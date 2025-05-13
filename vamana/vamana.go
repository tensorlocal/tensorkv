package vamana

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
)

const (
	DefaultAlpha      = 1.2
	DefaultL          = 100
	DefaultMaxDegree  = 64
	ConstructionAlpha = 1.0
)

type Graph struct {
	Nodes     []*Node
	MedoidID  int
	Alpha     float64
	MaxDegree int
	lock      sync.RWMutex
}

type Node struct {
	ID       int
	Vector   []float64
	OutEdges []int
}

type candidate struct {
	id      int
	dist    float64
	visited bool
}

type priorityQueue []*candidate

func (pq priorityQueue) Len() int           { return len(pq) }
func (pq priorityQueue) Less(i, j int) bool { return pq[i].dist < pq[j].dist }
func (pq priorityQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *priorityQueue) Push(x interface{}) {
	item := x.(*candidate)
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// 欧氏距离平方优化计算
func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func RobustPrune(p *Node, candidates []*Node, alpha float64, maxDegree int, vectors [][]float64) []int {
	if len(candidates) == 0 {
		return nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		return euclideanDistance(p.Vector, candidates[i].Vector) <
			euclideanDistance(p.Vector, candidates[j].Vector)
	})

	pruned := make([]int, 0, maxDegree)
	pruned = append(pruned, candidates[0].ID) // 保留最近邻

	maxAllowedDist := float64(math.MaxFloat32)
	if alpha > 1 {
		maxAllowedDist = euclideanDistance(p.Vector, candidates[0].Vector) * alpha
	}

	for _, candidate := range candidates[1:] {
		keep := true
		currentDist := euclideanDistance(p.Vector, candidate.Vector)

		// α cond check
		if alpha > 1 && currentDist > maxAllowedDist {
			continue
		}

		// existing check
		for _, existingID := range pruned {
			existingDist := euclideanDistance(vectors[existingID], candidate.Vector)
			if existingDist < currentDist {
				keep = false
				break
			}
		}

		if keep {
			pruned = append(pruned, candidate.ID)
			if len(pruned) >= maxDegree {
				break
			}
		}
	}

	return pruned
}

func GreedySearch(graph *Graph, startID int, target []float64, L int) []int {
	if len(graph.Nodes) == 1 {
		return []int{0} // 单节点情况下直接返回自身
	}

	visited := make(map[int]struct{})
	candidates := make(priorityQueue, 0)
	heap.Init(&candidates)

	// 从图结构中获取起始节点
	startNode := graph.Nodes[startID]
	initialDist := euclideanDistance(startNode.Vector, target)
	heap.Push(&candidates, &candidate{id: startID, dist: initialDist})

	result := make([]int, 0, L)

	for candidates.Len() > 0 && len(result) < L {
		current := heap.Pop(&candidates).(*candidate)
		if _, exists := visited[current.id]; exists {
			continue
		}
		visited[current.id] = struct{}{}
		result = append(result, current.id)

		currentNode := graph.Nodes[current.id]
		for _, neighborID := range currentNode.OutEdges { // 正确访问图的出边
			if _, exists := visited[neighborID]; !exists {
				neighborNode := graph.Nodes[neighborID]
				dist := euclideanDistance(neighborNode.Vector, target)
				heap.Push(&candidates, &candidate{id: neighborID, dist: dist})
			}
		}
	}

	return result
}

func BuildVamanaGraph(vectors [][]float64, alpha float64, maxDegree int) *Graph {
	n := len(vectors)
	if n == 0 {
		return nil
	}

	// 初始化图结构
	graph := &Graph{
		Nodes:     make([]*Node, n),
		Alpha:     alpha,
		MaxDegree: maxDegree,
	}

	// 初始化节点
	for i := range vectors {
		graph.Nodes[i] = &Node{
			ID:       i,
			Vector:   vectors[i],
			OutEdges: []int{},
		}
	}

	// 计算Medoid
	if n == 1 {
		graph.MedoidID = 0 // 单节点时直接设置为 0
		return graph       // 单节点情况直接返回，无需构建边
	} else {
		graph.MedoidID = computeMedoid(vectors)
	}

	// 两阶段构建
	for pass := 0; pass < 2; pass++ {
		currentAlpha := ConstructionAlpha
		if pass == 1 {
			currentAlpha = alpha
		}

		// 随机遍历顺序
		order := rand.Perm(n)
		for _, i := range order {
			node := graph.Nodes[i]

			// 执行贪婪搜索获取访问路径
			visitedIDs := GreedySearch(graph, graph.MedoidID, node.Vector, DefaultL)

			// 收集候选节点
			candidates := make([]*Node, 0, len(visitedIDs))
			for _, id := range visitedIDs {
				if id != i { // 不包括节点自身
					candidates = append(candidates, graph.Nodes[id])
				}
			}

			// 如果没有候选节点，至少添加medoid作为候选
			if len(candidates) == 0 && i != graph.MedoidID {
				candidates = append(candidates, graph.Nodes[graph.MedoidID])
			}

			// 执行RobustPrune
			newEdges := RobustPrune(node, candidates, currentAlpha, maxDegree, vectors)

			// 更新出边
			graph.lock.Lock()
			node.OutEdges = newEdges

			// 添加反向边并处理度数限制
			for _, neighborID := range newEdges {
				neighbor := graph.Nodes[neighborID]

				// 检查是否已经有这条边，避免重复添加
				hasEdge := false
				for _, e := range neighbor.OutEdges {
					if e == node.ID {
						hasEdge = true
						break
					}
				}

				if !hasEdge {
					neighbor.OutEdges = append(neighbor.OutEdges, node.ID)
				}

				if len(neighbor.OutEdges) > maxDegree {
					// 对邻居执行度数修剪
					neighborCandidates := make([]*Node, 0, len(neighbor.OutEdges))
					for _, e := range neighbor.OutEdges {
						neighborCandidates = append(neighborCandidates, graph.Nodes[e])
					}
					neighbor.OutEdges = RobustPrune(neighbor, neighborCandidates, currentAlpha, maxDegree, vectors)
				}
			}
			graph.lock.Unlock()
		}
	}

	// 检查图的连通性，确保所有节点都可从medoid到达
	ensureConnectivity(graph, vectors)

	return graph
}

// 确保图的连通性
func ensureConnectivity(graph *Graph, vectors [][]float64) {
	visited := make([]bool, len(graph.Nodes))
	queue := []int{graph.MedoidID}
	visited[graph.MedoidID] = true

	// 广度优先搜索标记可到达节点
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, neighbor := range graph.Nodes[current].OutEdges {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}

	// 连接未访问节点到已访问的最近节点
	for i, isVisited := range visited {
		if !isVisited {
			// 找到最近的已访问节点
			minDist := float64(math.MaxFloat64)
			closestVisited := -1

			for j, isJ := range visited {
				if isJ {
					dist := euclideanDistance(vectors[i], vectors[j])
					if dist < minDist {
						minDist = dist
						closestVisited = j
					}
				}
			}

			if closestVisited >= 0 {
				// 双向连接
				graph.Nodes[i].OutEdges = append(graph.Nodes[i].OutEdges, closestVisited)
				graph.Nodes[closestVisited].OutEdges = append(graph.Nodes[closestVisited].OutEdges, i)
			}
		}
	}
}

// 辅助函数：生成随机初始边
func randomEdges(totalNodes, numEdges int) []int {
	edges := make([]int, 0, numEdges)
	for i := 0; i < numEdges; i++ {
		edges = append(edges, rand.Intn(totalNodes))
	}
	return edges
}

// 辅助函数：计算Medoid节点
func computeMedoid(vectors [][]float64) int {
	type pair struct{ i, j int }
	distChan := make(chan float64, len(vectors)*(len(vectors)-1)/2)

	// 并行计算所有pair距离
	var wg sync.WaitGroup
	for i := 0; i < len(vectors); i++ {
		for j := i + 1; j < len(vectors); j++ {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				dist := euclideanDistance(vectors[i], vectors[j])
				distChan <- dist
			}(i, j)
		}
	}

	go func() {
		wg.Wait()
		close(distChan)
	}()

	// 计算总距离（此处简化为示例）
	minTotal := float64(math.MaxFloat32)
	medoid := 0
	for i := range vectors {
		total := float64(0)
		for j := range vectors {
			if i != j {
				total += euclideanDistance(vectors[i], vectors[j])
			}
		}
		if total < minTotal {
			minTotal = total
			medoid = i
		}
	}
	return medoid
}
