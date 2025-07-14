package vamana

import (
	_ "net/http/pprof"
	"testing"
)

func TestKMeans(t *testing.T) {
	t.Run("Basic clustering", func(t *testing.T) {
		vectors := [][]float64{
			{0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5}, // Cluster 1
			{10.0, 10.0}, {11.0, 11.0}, {10.5, 10.5}, // Cluster 2
		}

		centroids := KMeans(vectors, 2, 10)
		if len(centroids) != 2 {
			t.Errorf("Expected 2 centroids, got %d", len(centroids))
		}

		// Check that centroids are reasonable
		for _, centroid := range centroids {
			if len(centroid) != 2 {
				t.Errorf("Expected 2D centroid, got %dD", len(centroid))
			}
		}
	})

	t.Run("Empty vectors", func(t *testing.T) {
		vectors := [][]float64{}
		centroids := KMeans(vectors, 2, 10)
		if centroids != nil {
			t.Error("Expected nil centroids for empty input")
		}
	})

	t.Run("K larger than data", func(t *testing.T) {
		vectors := [][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		}
		centroids := KMeans(vectors, 5, 10)
		if len(centroids) != 2 {
			t.Errorf("Expected 2 centroids (capped), got %d", len(centroids))
		}
	})

	t.Run("Assign to L nearest", func(t *testing.T) {
		vectors := [][]float64{
			{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0},
		}
		centroids := [][]float64{
			{0.0, 0.0}, {3.0, 3.0},
		}

		assignments := assignToLNearest(vectors, centroids, 2)
		if len(assignments) != 2 {
			t.Errorf("Expected 2 assignment lists, got %d", len(assignments))
		}

		// Each point should be assigned to both centroids (L=2)
		for i, assignment := range assignments {
			if len(assignment) == 0 {
				t.Errorf("Assignment %d is empty", i)
			}
		}
	})
}
