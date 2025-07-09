package vamana

import (
	"os"
	"testing"
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

func TestDiskANN_BuildAndSearch_Integration(t *testing.T) {
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
