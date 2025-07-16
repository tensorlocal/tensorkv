package vamana

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"sync"
)

type DiskANN struct {
	indexPath         string
	fileHandle        *os.File
	MedoidID          int
	MaxDegree         int
	VectorDim         int
	nodeBlockSize     int64
	quantizer         Quantizer
	compressedVectors [][]byte
	lock              sync.RWMutex
}

func NewDiskANN(indexPath string, maxDegree, vectorDim int, quantizer Quantizer) (*DiskANN, error) {

	// FullVector (float64 = 8 bytes) + Neighbors (int = 4 bytes)

	nodeBlockSize := int64(vectorDim*8 + maxDegree*4)

	return &DiskANN{
		indexPath:     indexPath,
		MaxDegree:     maxDegree,
		VectorDim:     vectorDim,
		nodeBlockSize: nodeBlockSize,
		quantizer:     quantizer,
	}, nil
}

func (da *DiskANN) Build(vectors [][]float64, alpha float64, l int, k int) error {
	n := len(vectors)
	if n == 0 {
		return fmt.Errorf("cannot build index with zero vectors")
	}

	log.Println("Starting DiskANN build process...")

	allEdges := make(map[int]map[int]struct{}) // 使用 map 来存储全局的边关系

	VamanaGraph := BuildVamanaGraphForShard(vectors, alpha, da.MaxDegree)

	log.Printf("Finished build Graph %d", n)

	for _, node := range VamanaGraph.Nodes {
		if _, ok := allEdges[node.ID]; !ok {
			allEdges[node.ID] = make(map[int]struct{})
		}
		for _, neighborID := range node.OutEdges {
			allEdges[node.ID][neighborID] = struct{}{}
		}
	}

	log.Println("Writing final graph structure to disk...")

	da.MedoidID = computeMedoid(vectors)
	err := da.writeGraphToDisk(vectors, allEdges)
	if err != nil {
		return err
	}

	log.Println("Training quantizer and compressing vectors...")
	da.quantizer.Train(vectors)
	da.compressedVectors = make([][]byte, n)
	for i := range vectors {
		da.compressedVectors[i] = da.quantizer.Compress(vectors[i])
	}

	log.Println("DiskANN build completed successfully.")
	
	return nil
}

func (da *DiskANN) Search(query []float64, k int, L int, beamWidth int) ([]int, error) {
	da.lock.RLock()
	defer da.lock.RUnlock()

	if da.fileHandle == nil {
		var err error
		da.fileHandle, err = os.OpenFile(da.indexPath, os.O_RDONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("failed to open index file: %w", err)
		}
	}

	// 候选队列 - 使用近似距离，按距离从小到大排序（最小堆）
	type candidateItem struct {
		id   int
		dist float64
	}
	candidates := make([]candidateItem, 0)

	// 结果集 - 存储精确距离的结果
	type resultItem struct {
		id   int
		dist float64
	}
	results := make([]resultItem, 0)

	visited := make(map[int]struct{})

	// 从 Medoid 开始搜索
	startID := da.MedoidID
	visited[startID] = struct{}{}
	approxDist := da.quantizer.ApproximateDistance(query, da.compressedVectors[startID])
	candidates = append(candidates, candidateItem{id: startID, dist: approxDist})

	for len(candidates) > 0 {
		// 按距离排序候选队列，取距离最小的
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].dist < candidates[j].dist
		})

		// 提取 beamWidth 个最近的候选者
		var beam []candidateItem
		extractCount := beamWidth
		if extractCount > len(candidates) {
			extractCount = len(candidates)
		}

		beam = candidates[:extractCount]
		candidates = candidates[extractCount:]

		// 从磁盘批量读取这些节点的数据
		beamCandidates := make([]*candidate, len(beam))
		for i, item := range beam {
			beamCandidates[i] = &candidate{id: item.id, dist: item.dist}
		}

		fullNodes, err := da.readNodesFromDisk(beamCandidates)
		if err != nil {
			return nil, err
		}

		for _, node := range fullNodes {
			// 计算精确距离
			preciseDist := euclideanDistanceUnsafe(query, node.Vector)

			// 添加到结果集
			results = append(results, resultItem{id: node.ID, dist: preciseDist})

			// 探索邻居
			for _, neighborID := range node.OutEdges {
				if _, exists := visited[neighborID]; !exists {
					visited[neighborID] = struct{}{}
					approxDist := da.quantizer.ApproximateDistance(query, da.compressedVectors[neighborID])
					candidates = append(candidates, candidateItem{id: neighborID, dist: approxDist})
				}
			}
		}

		// 保持结果集大小不超过L，按精确距离排序并保留最近的L个
		if len(results) > L {
			sort.Slice(results, func(i, j int) bool {
				return results[i].dist < results[j].dist
			})
			results = results[:L]
		}

		// 早期终止条件：如果候选队列为空或者候选队列中最近的点比结果集中最远的点还要远
		if len(candidates) > 0 && len(results) >= L {
			sort.Slice(results, func(i, j int) bool {
				return results[i].dist < results[j].dist
			})
			farthestResultDist := results[len(results)-1].dist

			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].dist < candidates[j].dist
			})
			closestCandidateDist := candidates[0].dist

			// 由于使用的是近似距离，这里需要一个容忍度
			if closestCandidateDist > farthestResultDist*1.5 {
				break
			}
		}
	}

	// 最终排序并返回 top-k 结果
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	finalResults := make([]int, 0, k)
	for i := 0; i < k && i < len(results); i++ {
		finalResults = append(finalResults, results[i].id)
	}

	return finalResults, nil
}

func (da *DiskANN) writeGraphToDisk(vectors [][]float64, allEdges map[int]map[int]struct{}) error {
	file, err := os.Create(da.indexPath)
	if err != nil {
		return fmt.Errorf("failed to create index file: %w", err)
	}
	defer file.Close()

	for i, vec := range vectors {
		offset := int64(i) * da.nodeBlockSize
		file.Seek(offset, io.SeekStart)

		// 写入全精度向量
		buf := make([]byte, da.VectorDim*8)
		for j, val := range vec {
			binary.LittleEndian.PutUint64(buf[j*8:], math.Float64bits(val))
		}
		_, err := file.Write(buf)
		if err != nil {
			return err
		}

		// 初始化邻居数组为 -1 (而不是 0)
		neighbors := make([]int32, da.MaxDegree)
		for j := range neighbors {
			neighbors[j] = -1 // 使用 -1 作为无效邻居标记
		}

		nodeEdges, ok := allEdges[i]
		if ok {
			idx := 0
			for neighborID := range nodeEdges {
				if idx >= da.MaxDegree {
					break
				}
				neighbors[idx] = int32(neighborID)
				idx++
			}
		}

		neighborBuf := make([]byte, da.MaxDegree*4)
		for j, id := range neighbors {
			binary.LittleEndian.PutUint32(neighborBuf[j*4:], uint32(id))
		}
		_, err = file.Write(neighborBuf)
		if err != nil {
			return err
		}
	}
	return nil
}

func (da *DiskANN) readNodesFromDisk(nodesToRead []*candidate) ([]*Node, error) {
	fullNodes := make([]*Node, 0, len(nodesToRead))
	buf := make([]byte, da.nodeBlockSize)

	for _, c := range nodesToRead {
		offset := int64(c.id) * da.nodeBlockSize
		_, err := da.fileHandle.ReadAt(buf, offset)
		if err != nil {
			return nil, fmt.Errorf("failed to read node %d: %w", c.id, err)
		}

		// 解析全精度向量
		vec := make([]float64, da.VectorDim)
		for i := 0; i < da.VectorDim; i++ {
			bits := binary.LittleEndian.Uint64(buf[i*8 : (i+1)*8])
			vec[i] = math.Float64frombits(bits)
		}

		// 解析邻居列表
		edges := make([]int, 0, da.MaxDegree)
		edgeOffset := da.VectorDim * 8
		for i := 0; i < da.MaxDegree; i++ {
			id := int(int32(binary.LittleEndian.Uint32(buf[edgeOffset+i*4 : edgeOffset+(i+1)*4])))
			if id != -1 { // 使用 -1 作为无效邻居标记
				edges = append(edges, id)
			}
		}

		fullNodes = append(fullNodes, &Node{
			ID:       c.id,
			Vector:   vec,
			OutEdges: edges,
		})
	}
	return fullNodes, nil
}

func (da *DiskANN) getPreciseDist(query []float64, nodeID int) (float64, error) {
	nodes, err := da.readNodesFromDisk([]*candidate{{id: nodeID}})
	if err != nil {
		return 0, err
	}
	return euclideanDistanceUnsafe(query, nodes[0].Vector), nil
}
