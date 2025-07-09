package vamana

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

//todo: need more for production

// KMeans 实现K-Means聚类算法
func KMeans(vectors [][]float64, k, maxIterations int) [][]float64 {
	if len(vectors) == 0 {
		return nil
	}
	if k > len(vectors) {
		k = len(vectors)
	}

	dim := len(vectors[0])
	centroids := make([][]float64, k)
	// 随机选择初始质心
	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(vectors))
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], vectors[indices[i]])
	}

	for iter := 0; iter < maxIterations; iter++ {
		clusters := make([][]int, k)
		for i, vec := range vectors {
			minDist := math.MaxFloat64
			closestCentroid := 0
			for j, centroid := range centroids {
				dist := euclideanDistanceSquared(vec, centroid)
				if dist < minDist {
					minDist = dist
					closestCentroid = j
				}
			}
			clusters[closestCentroid] = append(clusters[closestCentroid], i)
		}

		converged := true
		for i := 0; i < k; i++ {
			if len(clusters[i]) == 0 {
				// 重新随机选择质心
				randIndex := rand.Intn(len(vectors))
				centroids[i] = make([]float64, dim)
				copy(centroids[i], vectors[randIndex])
				converged = false
				continue
			}

			newCentroid := make([]float64, dim)
			for _, vectorIndex := range clusters[i] {
				for d := 0; d < dim; d++ {
					newCentroid[d] += vectors[vectorIndex][d]
				}
			}
			for d := 0; d < dim; d++ {
				newCentroid[d] /= float64(len(clusters[i]))
			}

			if euclideanDistanceSquared(centroids[i], newCentroid) > 1e-10 {
				converged = false
			}
			centroids[i] = newCentroid
		}

		if converged {
			break
		}
	}

	return centroids
}

// assignToLNearest 将每个点分配到l个最近的聚类中心，以创建重叠分片
func assignToLNearest(vectors [][]float64, centroids [][]float64, l int) [][]int {
	assignments := make([][]int, len(centroids))

	for i, vec := range vectors {
		type centroidDist struct {
			index int
			dist  float64
		}
		dists := make([]centroidDist, len(centroids))
		for j, center := range centroids {
			dists[j] = centroidDist{index: j, dist: euclideanDistance(vec, center)}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })

		for j := 0; j < l && j < len(dists); j++ {
			centroidIdx := dists[j].index
			assignments[centroidIdx] = append(assignments[centroidIdx], i)
		}
	}
	return assignments
}
