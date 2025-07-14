package vamana

import "testing"

var result float64

func BenchmarkEuclidean1024d(b *testing.B) {
	vecA := createTestVectors(1, 1024)
	vecB := createTestVectors(1, 1024)
	var r float64
	for i := 0; i < b.N; i++ {
		r = euclideanDistance(vecA[0], vecB[0])

	}
	result = r

}
