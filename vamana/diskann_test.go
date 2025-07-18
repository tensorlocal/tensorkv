package vamana

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func TestNewDiskANN(t *testing.T) {
	indexPath := "/tmp/test.index"
	maxDegree := 32
	vectorDim := 128
	quantizer := NewProductQuantizer(16, 256, true)

	da, err := NewDiskANN(indexPath, maxDegree, vectorDim, quantizer)

	if err != nil {
		t.Fatalf("NewDiskANN() returned an unexpected error: %v", err)
	}
	if da == nil {
		t.Fatal("NewDiskANN() returned a nil DiskANN instance")
	}
	if da.indexPath != indexPath {
		t.Errorf("Expected indexPath to be %s, got %s", indexPath, da.indexPath)
	}
	if da.MaxDegree != maxDegree {
		t.Errorf("Expected MaxDegree to be %d, got %d", maxDegree, da.MaxDegree)
	}
	if da.VectorDim != vectorDim {
		t.Errorf("Expected VectorDim to be %d, got %d", vectorDim, da.VectorDim)
	}
	expectedBlockSize := int64(vectorDim*8 + maxDegree*4)
	if da.nodeBlockSize != expectedBlockSize {
		t.Errorf("Expected nodeBlockSize to be %d, got %d", expectedBlockSize, da.nodeBlockSize)
	}
	if da.quantizer == nil {
		t.Error("Expected quantizer to be non-nil")
	}
}

func TestDiskANN_BuildAndSearch_Single(t *testing.T) {
	indexPath := createTempFile(t)
	vectorDim := 2
	maxDegree := 4
	mockQ := NewProductQuantizer(2, 4, true)

	// Sample data where vectors are clearly distinct.
	vectors := [][]float64{
		{0.0, 0.0},     // 0
		{1.0, 1.0},     // 1
		{100.0, 100.0}, // 2
		{11.0, 11.0},   // 3
		{11.1, 11.1},   // 4
		{21.0, 21.0},   // 5
	}

	da, err := NewDiskANN(indexPath, maxDegree, vectorDim, mockQ)
	if err != nil {
		t.Fatalf("Failed to create DiskANN instance: %v", err)
	}

	// Build parameters
	// Using high alpha for a dense graph in this small dataset
	alpha := 1.0
	// L (build list size) and K (shards)
	l := 4
	k := 2

	// Build the index
	err = da.Build(vectors, alpha, l, k)
	if err != nil {
		t.Fatalf("Build() failed with error: %v", err)
	}

	// Verify the index file was created and is not empty
	fileInfo, err := os.Stat(indexPath)
	if os.IsNotExist(err) {
		t.Fatalf("Index file was not created at %s", indexPath)
	}
	if fileInfo.Size() == 0 {
		t.Fatal("Index file is empty after build")
	}

	// --- Search Test ---
	// Query vector close to {1.0, 1.0}
	query := []float64{1.1, 1.1}
	// Search for the top 2 nearest neighbors
	kSearch := 2
	// L (search list size) and beamWidth
	lSearch := 6
	beamWidth := 4

	results, err := da.Search(query, kSearch, lSearch, beamWidth)
	if err != nil {
		t.Fatalf("Search() failed with error: %v", err)
	}

	if len(results) != kSearch {
		t.Fatalf("Expected %d results, but got %d", kSearch, len(results))
	}

	// The closest vector should be index 1 ({1.0, 1.0})
	expectedFirstResult := 1
	for _, x := range results {
		t.Log("results: ", x)
	}
	if results[0] != expectedFirstResult {
		t.Errorf("Expected the first result to be %d, but got %d", expectedFirstResult, results[0])
	}

	// The second closest vector should be index 0 ({0.0, 0.0})
	expectedSecondResult := 0
	if results[1] != expectedSecondResult {
		t.Errorf("Expected the second result to be %d, but got %d", expectedSecondResult, results[1])
	}

	t.Logf("Search results for query %v: %v", query, results)
}

func TestDiskANN_Search(t *testing.T) {
	// Create a predictable dataset
	vectors := [][]float64{
		{0.0, 0.0},   // 0
		{1.0, 1.0},   // 1
		{2.0, 2.0},   // 2
		{10.0, 10.0}, // 3
		{11.0, 11.0}, // 4
		{12.0, 12.0}, // 5
	}

	indexPath := createTempFile(t)
	da, err := NewDiskANN(indexPath, 8, 2, NewProductQuantizer(2, 4, false))
	if err != nil {
		t.Fatalf("Failed to create DiskANN: %v", err)
	}

	err = da.Build(vectors, 1.0, 4, 2)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	tests := []struct {
		name      string
		query     []float64
		k         int
		L         int
		beamWidth int
		wantErr   bool
		validate  func(*testing.T, []int)
	}{
		{
			name:      "Query close to origin",
			query:     []float64{0.1, 0.1},
			k:         2,
			L:         6,
			beamWidth: 4,
			wantErr:   false,
			validate: func(t *testing.T, results []int) {
				if len(results) != 2 {
					t.Errorf("Expected 2 results, got %d", len(results))
				}
				// Should find points 0 and 1 as closest
				if results[0] != 0 {
					t.Errorf("Expected first result to be 0, got %d", results[0])
				}
			},
		},
		{
			name:      "Query close to cluster 2",
			query:     []float64{10.5, 10.5},
			k:         3,
			L:         6,
			beamWidth: 4,
			wantErr:   false,
			validate: func(t *testing.T, results []int) {
				if len(results) != 3 {
					t.Errorf("Expected 3 results, got %d", len(results))
				}
				// Should find points in the 10-12 range
				found := make(map[int]bool)
				for _, r := range results {
					found[r] = true
				}
				expectedInResults := []int{3, 4, 5}
				for _, expected := range expectedInResults {
					if !found[expected] {
						t.Errorf("Expected to find point %d in results", expected)
					}
				}
			},
		},
		{
			name:      "Single result",
			query:     []float64{1.1, 1.1},
			k:         1,
			L:         6,
			beamWidth: 4,
			wantErr:   false,
			validate: func(t *testing.T, results []int) {
				if len(results) != 1 {
					t.Errorf("Expected 1 result, got %d", len(results))
				}
				if results[0] != 1 {
					t.Errorf("Expected result to be 1, got %d", results[0])
				}
			},
		},
		{
			name:      "Large k",
			query:     []float64{5.0, 5.0},
			k:         10, // More than available vectors
			L:         6,
			beamWidth: 4,
			wantErr:   false,
			validate: func(t *testing.T, results []int) {
				if len(results) > 6 {
					t.Errorf("Expected at most 6 results, got %d", len(results))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := da.Search(tt.query, tt.k, tt.L, tt.beamWidth)
			if (err != nil) != tt.wantErr {
				t.Errorf("Search() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.validate != nil {
				tt.validate(t, results)
			}
		})
	}
}

func TestDiskANNCorrectness(t *testing.T) {
	go func() {
		log.Println(http.ListenAndServe(":6060", nil))
	}()
	testConfigs := []TestConfig{
		// 小规模测试
		{Dimension: 8, DataSize: 10, K: 10, L: 50, Alpha: 1.2, MaxDegree: 4, PQM: 8, PQKs: 256},
		{Dimension: 64, DataSize: 200, K: 10, L: 50, Alpha: 1.2, MaxDegree: 32, PQM: 8, PQKs: 256},
		{Dimension: 128, DataSize: 500, K: 10, L: 50, Alpha: 1.2, MaxDegree: 32, PQM: 16, PQKs: 256},

		// 中等规模测试
		{Dimension: 128, DataSize: 1000, K: 20, L: 100, Alpha: 1.2, MaxDegree: 70, PQM: 64, PQKs: 256},
		{Dimension: 128, DataSize: 2000, K: 20, L: 100, Alpha: 1.2, MaxDegree: 70, PQM: 64, PQKs: 256},
		{Dimension: 128, DataSize: 3000, K: 20, L: 100, Alpha: 1.2, MaxDegree: 70, PQM: 64, PQKs: 256},
		{Dimension: 128, DataSize: 4000, K: 20, L: 100, Alpha: 1.2, MaxDegree: 70, PQM: 64, PQKs: 256},
		{Dimension: 128, DataSize: 5000, K: 20, L: 100, Alpha: 1.2, MaxDegree: 70, PQM: 64, PQKs: 256},

		// 大规模测试
		{Dimension: 1024, DataSize: 5000, K: 50, L: 200, Alpha: 1.2, MaxDegree: 128, PQM: 64, PQKs: 256},
		{Dimension: 1024, DataSize: 10000, K: 50, L: 200, Alpha: 1.2, MaxDegree: 128, PQM: 4, PQKs: 256},

		// 极端情况测试
		{Dimension: 2, DataSize: 50, K: 5, L: 20, Alpha: 1.2, MaxDegree: 16, PQM: 2, PQKs: 256},
		{Dimension: 20480, DataSize: 10, K: 10, L: 100, Alpha: 1.2, MaxDegree: 64, PQM: 20480, PQKs: 256},
	}

	fmt.Println("开始 DiskANN 正确性测试...")

	var passedTests, totalTests int

	for i, config := range testConfigs {
		t.Run(fmt.Sprintf("Test_%d_Dim%d_Size%d", i+1, config.Dimension, config.DataSize), func(t *testing.T) {
			totalTests++

			fmt.Printf("测试 %d: 维度=%d, 数据量=%d, K=%d, L=%d\n",
				i+1, config.Dimension, config.DataSize, config.K, config.L)

			result := testDiskANN(t, config)

			// 输出结果
			fmt.Printf("  构建时间: %v\n", result.BuildTime)
			fmt.Printf("  搜索时间: %v\n", result.SearchTime)
			fmt.Printf("  索引大小: %d KB\n", result.IndexSize/1024)
			fmt.Printf("  召回率: %.3f\n", result.Recall)
			fmt.Printf("  平均距离: %.3f\n", result.AvgDistance)

			if result.Success {
				fmt.Printf("  状态: ✓ 通过\n")
				passedTests++
			} else {
				fmt.Printf("  状态: ✗ 失败\n")
				for _, err := range result.Errors {
					fmt.Printf("    错误: %s\n", err)
					t.Errorf("测试失败: %s", err)
				}
			}

			// 基本正确性检查
			if result.Recall < 0.3 {
				t.Errorf("召回率过低: %.3f < 0.3", result.Recall)
			}

			if result.BuildTime > time.Minute*5 {
				t.Errorf("构建时间过长: %v > 5分钟", result.BuildTime)
			}

			if result.SearchTime > time.Second {
				t.Errorf("搜索时间过长: %v > 1秒", result.SearchTime)
			}
		})
	}

	fmt.Printf("\n总体测试结果: %d/%d 通过 (%.1f%%)\n 调用prune : %d",
		passedTests, totalTests, float64(passedTests)/float64(totalTests)*100, totalPruneC)
}

func testDiskANN(t *testing.T, config TestConfig) TestResult {
	result := TestResult{
		Config: config,
		Errors: make([]string, 0),
	}

	// 创建临时目录
	tempDir := t.TempDir()
	indexPath := filepath.Join(tempDir, "test.idx")

	// 生成测试数据
	vectors := createTestVectors(config.DataSize, config.Dimension)

	// 创建量化器
	quantizer := NewProductQuantizer(config.PQM, config.PQKs, false)

	// 创建 DiskANN 实例
	diskann, err := NewDiskANN(indexPath, config.MaxDegree, config.Dimension, quantizer)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("创建 DiskANN 失败: %v", err))
		return result
	}

	// 构建索引
	startTime := time.Now()
	err = diskann.Build(vectors, config.Alpha, config.L, config.K)
	result.BuildTime = time.Since(startTime)

	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("构建索引失败: %v", err))
		return result
	}

	// 检查索引文件大小
	if info, err := os.Stat(indexPath); err == nil {
		result.IndexSize = info.Size()
	}

	// 生成查询向量
	numQueries := min(10, config.DataSize/10)
	queries := createTestVectors(numQueries, config.Dimension)

	var totalRecall float64
	var totalAvgDist float64
	var totalSearchTime time.Duration

	// 执行搜索测试
	for i, query := range queries {
		// 暴力搜索作为真实结果
		groundTruth := bruteForceSearch(vectors, query, config.K)

		// DiskANN 搜索
		searchStart := time.Now()
		searchResult, err := diskann.Search(query, config.K, config.L*2, 4)
		searchTime := time.Since(searchStart)
		totalSearchTime += searchTime

		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("查询 %d 失败: %v", i, err))
			continue
		}

		// 计算召回率
		recall := calculateRecall(groundTruth, searchResult)
		totalRecall += recall

		// 计算平均距离
		avgDist := calculateAverageDistance(vectors, query, searchResult)
		totalAvgDist += avgDist

		// 详细验证
		if recall < 0.5 {
			result.Errors = append(result.Errors, fmt.Sprintf("查询 %d 召回率过低: %.2f", i, recall))
		}
	}

	result.Recall = totalRecall / float64(numQueries)
	result.AvgDistance = totalAvgDist / float64(numQueries)
	result.SearchTime = totalSearchTime / time.Duration(numQueries)
	result.Success = len(result.Errors) == 0

	return result
}

func calculateRecall(groundTruth, searchResult []int) float64 {
	if len(groundTruth) == 0 || len(searchResult) == 0 {
		return 0.0
	}

	truthSet := make(map[int]bool)
	for _, id := range groundTruth {
		truthSet[id] = true
	}

	matches := 0
	for _, id := range searchResult {
		if truthSet[id] {
			matches++
		}
	}

	return float64(matches) / float64(len(groundTruth))
}

func generateClusteredVectors(dimension, count, numClusters int, seed int64) [][]float64 {
	rand.Seed(seed)
	vectors := make([][]float64, count)

	// 生成聚类中心
	centers := make([][]float64, numClusters)
	for i := range centers {
		centers[i] = make([]float64, dimension)
		for j := range centers[i] {
			centers[i][j] = rand.NormFloat64() * 5 // 聚类中心分布更广
		}
	}

	// 为每个向量分配到聚类
	for i := range vectors {
		clusterIdx := rand.Intn(numClusters)
		vectors[i] = make([]float64, dimension)
		for j := range vectors[i] {
			vectors[i][j] = centers[clusterIdx][j] + rand.NormFloat64()*0.5 // 聚类内噪声
		}
	}

	return vectors
}

func bruteForceSearch(vectors [][]float64, query []float64, k int) []int {
	type result struct {
		id   int
		dist float64
	}

	results := make([]result, len(vectors))
	for i, vec := range vectors {
		results[i] = result{id: i, dist: euclideanDistance(query, vec)}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	topK := make([]int, k)
	for i := 0; i < k && i < len(results); i++ {
		topK[i] = results[i].id
	}

	return topK
}

// calculateAverageDistance 计算平均距离
func calculateAverageDistance(vectors [][]float64, query []float64, resultIDs []int) float64 {
	if len(resultIDs) == 0 {
		return 0.0
	}

	var totalDist float64
	for _, id := range resultIDs {
		totalDist += math.Sqrt(euclideanDistance(query, vectors[id]))
	}

	return totalDist / float64(len(resultIDs))
}

func createTempFile(t *testing.T) string {
	t.Helper()
	tmpFile, err := os.CreateTemp("", "diskann-*.index")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	tmpFile.Close() // Close the file so the library can open it
	t.Cleanup(func() {
		os.Remove(tmpFile.Name())
	})
	return tmpFile.Name()
}

func createTestVectors(n, dim int) [][]float64 {
	rand.Seed(42) // Fixed seed for reproducible tests
	vectors := make([][]float64, n)
	for i := range vectors {
		vectors[i] = make([]float64, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.NormFloat64()
		}
	}
	return vectors
}

type TestConfig struct {
	Dimension int
	DataSize  int
	K         int
	L         int
	Alpha     float64
	MaxDegree int
	PQM       int
	PQKs      int
}

// TestResult 测试结果
type TestResult struct {
	Config      TestConfig
	Recall      float64
	AvgDistance float64
	BuildTime   time.Duration
	SearchTime  time.Duration
	IndexSize   int64
	Errors      []string
	Success     bool
}
