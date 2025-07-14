package vamana

import (
	"log"
	"math"
	"sync"
	"time"
)

type ProductQuantizer struct {
	m         int           // 子向量数量
	ks        int           // 每个子码本的质心数
	d         int           // 向量维度
	ds        int           // 子向量维度
	codebooks [][][]float64 // 码本：[m][ks][ds]
	verbose   bool
	seed      int64
}

// NewProductQuantizer 创建一个新的 ProductQuantizer 实例
// m: 子向量数量。D 必须能被 m 整除。
// ks: 每个子码本的质心数。为了用 byte 存储，通常为 256。
func NewProductQuantizer(m, ks int, verbose bool) *ProductQuantizer {
	if ks > 256 {
		log.Printf("Warning: ks > 256. Compression will require more than 1 byte per sub-quantizer.")
	}
	return &ProductQuantizer{
		m:       m,
		ks:      ks,
		verbose: verbose,
		seed:    time.Now().UnixNano(),
	}
}

// Train 并发地训练所有子量化器的码本
func (pq *ProductQuantizer) Train(vectors [][]float64) {
	if len(vectors) == 0 {
		log.Fatal("Training vectors cannot be empty")
	}
	pq.d = len(vectors[0])
	if pq.d%pq.m != 0 {
		log.Fatalf("Vector dimension %d is not divisible by m=%d", pq.d, pq.m)
	}
	pq.ds = pq.d / pq.m
	pq.codebooks = make([][][]float64, pq.m)

	var wg sync.WaitGroup
	if pq.verbose {
		log.Printf("Starting training with %d vectors of dimension %d", len(vectors), pq.d)
		log.Printf("PQ parameters: M=%d, Ks=%d, Ds=%d", pq.m, pq.ks, pq.ds)
	}

	for i := 0; i < pq.m; i++ {
		wg.Add(1)
		go func(subQuantizerIndex int) {
			defer wg.Done()
			// 1. 为当前子量化器准备训练数据
			subVectors := make([][]float64, len(vectors))
			for j, vec := range vectors {
				start := subQuantizerIndex * pq.ds
				end := start + pq.ds
				// 创建子向量的副本以保证数据安全
				subVec := make([]float64, pq.ds)
				copy(subVec, vec[start:end])
				subVectors[j] = subVec
			}

			// 2. 对子向量运行 K-Means
			if pq.verbose {
				log.Printf("Training sub-quantizer %d...", subQuantizerIndex)
			}
			// 使用确定性的随机种子进行K-Means
			// localRand := rand.New(rand.NewSource(pq.seed + int64(subQuantizerIndex)))
			centroids := KMeans(subVectors, pq.ks, 25) // 修复：传入正确的参数
			pq.codebooks[subQuantizerIndex] = centroids
			if pq.verbose {
				log.Printf("Sub-quantizer %d trained.", subQuantizerIndex)
			}
		}(i)
	}
	wg.Wait()
	log.Println("Training complete.")
}

// Compress 将一个向量压缩为 M 个字节
func (pq *ProductQuantizer) Compress(vector []float64) []byte {
	if len(vector) != pq.d {
		log.Fatalf("Input vector dimension %d does not match trained dimension %d", len(vector), pq.d)
	}
	compressed := make([]byte, pq.m)
	for i := 0; i < pq.m; i++ {
		start := i * pq.ds // 修复：添加缺失的乘法运算符
		end := start + pq.ds
		subVector := vector[start:end]
		// 找到最近的质心ID
		bestCentroidID := -1
		minDistSq := math.MaxFloat64
		for j, centroid := range pq.codebooks[i] {
			distSq := euclideanDistanceUnsafe(subVector, centroid) // 修复：使用平方距离进行比较
			if distSq < minDistSq {
				minDistSq = distSq
				bestCentroidID = j
			}
		}
		compressed[i] = byte(bestCentroidID)
	}
	return compressed
}

// ApproximateDistance 使用非对称距离计算 (ADC) 来估算距离
// 这是性能的关键：预计算距离表 + 查表
func (pq *ProductQuantizer) ApproximateDistance(query []float64, compressed []byte) float64 {
	if len(query) != pq.d {
		log.Fatalf("Query vector dimension %d does not match trained dimension %d", len(query), pq.d)
	}
	// 1. 预计算距离查找表 (distance table)
	// distTable[i][j] = distance(query*sub_vector_i, centroid_j_of_codebook_i)^2
	distTable := make([][]float64, pq.m)
	for i := 0; i < pq.m; i++ {
		distTable[i] = make([]float64, pq.ks)
		start := i * pq.ds
		end := start + pq.ds
		querySubVector := query[start:end]

		// 修复：检查码本是否已初始化
		if len(pq.codebooks[i]) == 0 {
			log.Fatalf("Codebook %d is empty. Make sure to call Train() first.", i)
		}

		for j := 0; j < pq.ks && j < len(pq.codebooks[i]); j++ { // 修复：添加边界检查
			distTable[i][j] = euclideanDistanceUnsafe(querySubVector, pq.codebooks[i][j])
		}
	}

	// 2. 查表并累加距离
	var totalDistSq float64
	for i := 0; i < pq.m; i++ {
		centroidID := int(compressed[i])    // 修复：转换为int类型
		if centroidID < len(distTable[i]) { // 修复：添加边界检查
			totalDistSq += distTable[i][centroidID]
		}
	}
	return math.Sqrt(totalDistSq) // 返回欧氏距离，而不是距离平方
}

// CompressedVectorSize 返回压缩后的大小
func (pq *ProductQuantizer) CompressedVectorSize() int {
	return pq.m
}
