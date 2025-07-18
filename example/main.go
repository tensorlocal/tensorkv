// main.go
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	// 假设您已将上面的代码保存在 "vamana" 目录下
	// 请根据您的项目结构调整此导入路径
	"tensorkv/vamana"
)

func main() {
	// --- 1. 参数和环境设置 ---
	rand.Seed(time.Now().UnixNano())

	const (
		numVectors    = 2000               // 数据集中的向量总数
		vectorDim     = 16                 // 每个向量的维度
		indexPath     = "my_diskann.index" // 磁盘索引文件名
		maxDegree     = 32                 // Vamana图的最大出度
		kNeighbors    = 5                  // 我们想要查找的最近邻数量
		quantizerSize = 32                 // 压缩向量的大小（字节）
	)

	// 在程序结束时自动删除生成的索引文件
	defer os.Remove(indexPath)

	log.Println("--- DiskANN Usage Example ---")
	log.Printf("Dataset: %d vectors, %d dimensions", numVectors, vectorDim)

	// --- 2. 生成随机测试数据 ---
	log.Println("Step 1: Generating random data...")
	vectors := generateRandomData(numVectors, vectorDim)
	log.Println("Data generation complete.")

	// --- 3. 初始化 DiskANN 和量化器 ---
	log.Println("Step 2: Initializing DiskANN...")

	// 使用一个简化的量化器作为示例
	quantizer := vamana.NewProductQuantizer(16, 256, true)

	// 创建 DiskANN 实例
	diskann, err := vamana.NewDiskANN(indexPath, maxDegree, vectorDim, quantizer)
	if err != nil {
		log.Fatalf("Failed to create DiskANN instance: %v", err)
	}
	log.Println("DiskANN instance created.")

	// --- 4. 构建索引 ---
	log.Println("Step 3: Building the index using K-Means sharding...")
	// 构建参数
	const (
		buildAlpha = 1.2 // Vamana 构建参数 alpha
		lOverlap   = 2   // 每个点分配到 l 个最近的簇，以创建重叠分片
		kClusters  = 10  // K-Means 的簇数量
	)

	err = diskann.Build(vectors, buildAlpha, lOverlap, kClusters)
	if err != nil {
		log.Fatalf("Failed to build index: %v", err)
	}
	log.Println("Index build process finished and saved to disk.")

	// --- 5. 执行搜索 ---
	log.Println("Step 4: Performing a search...")
	// 创建一个查询向量（这里我们使用数据集中的第一个向量作为查询）
	queryVector := vectors[0]

	// 搜索参数
	const (
		searchL   = 50 // 搜索列表大小
		beamWidth = 4  // Beam Search 的宽度
	)

	log.Printf("Querying for %d nearest neighbors of vector 0...", kNeighbors)

	// 调用 Beam Search
	neighborIDs, err := diskann.Search(queryVector, kNeighbors, searchL, beamWidth)
	if err != nil {
		log.Fatalf("Failed to perform search: %v", err)
	}

	// --- 6. 打印结果 ---
	log.Println("--- Search Results ---")
	fmt.Printf("Query Vector ID: 0\n")
	fmt.Printf("Found %d nearest neighbor IDs: %v\n", kNeighbors, neighborIDs)
	// 验证：结果的第一个通常是它自身（ID为0）
	if len(neighborIDs) > 0 && neighborIDs[0] == 0 {
		log.Println("Verification successful: The nearest neighbor is the vector itself.")
	}
	log.Println("----------------------")

	log.Println("Example finished. Index file will be cleaned up.")
}

// generateRandomData 是一个辅助函数，用于创建指定数量和维度的随机向量
func generateRandomData(num, dim int) [][]float32 {
	data := make([][]float32, num)
	for i := 0; i < num; i++ {
		vector := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vector[j] = rand.Float64()
		}
		data[i] = vector
	}
	return data
}
