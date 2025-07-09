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
	DefaultBeamWidth  = 4
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
func (pq priorityQueue) Less(i, j int) bool { return pq[i].dist > pq[j].dist }
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

func RobustPrune(p *Node, candidates []*Node, alpha float64, maxDegree int) []int {
	if len(candidates) == 0 {
		return nil
	}

	// Create a copy to avoid modifying the original slice of candidates
	sortedCandidates := make([]*Node, len(candidates))
	copy(sortedCandidates, candidates)

	sort.Slice(sortedCandidates, func(i, j int) bool {
		return euclideanDistance(p.Vector, sortedCandidates[i].Vector) <
			euclideanDistance(p.Vector, sortedCandidates[j].Vector)
	})

	pruned := make([]int, 0, maxDegree)

	for _, candidate := range sortedCandidates {
		if len(pruned) >= maxDegree {
			break
		}

		keep := true
		distToCandidate := euclideanDistance(p.Vector, candidate.Vector)

		for _, existingID := range pruned {
			// Need the full node to get the vector
			var existingNode *Node
			for _, n := range sortedCandidates {
				if n.ID == existingID {
					existingNode = n
					break
				}
			}
			// This part is tricky without a map from ID to Node.
			// Let's assume we can find it for now. A map would be more efficient.
			if existingNode != nil {
				distBetweenSelected := euclideanDistance(existingNode.Vector, candidate.Vector)
				if alpha*distBetweenSelected < distToCandidate {
					keep = false
					break
				}
			}
		}

		if keep {
			pruned = append(pruned, candidate.ID)
		}
	}

	return pruned
}

func GreedySearch(graph *Graph, startID int, target []float64, L int) ([]int, []*Node) {
	visited := make(map[int]struct{})
	candidates := &priorityQueue{}
	heap.Init(candidates)

	// Add start node to candidates
	dist := euclideanDistance(graph.Nodes[startID].Vector, target)
	heap.Push(candidates, &candidate{id: startID, dist: dist})

	resultPool := &priorityQueue{}
	heap.Init(resultPool)
	heap.Push(resultPool, &candidate{id: startID, dist: dist})

	visitedNodesForPruning := []*Node{graph.Nodes[startID]}

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*candidate)

		if _, exists := visited[current.id]; exists {
			continue
		}
		visited[current.id] = struct{}{}

		// Check termination condition
		if resultPool.Len() >= L {
			farthestResult := (*resultPool)[0]
			if current.dist > farthestResult.dist && resultPool.Len() >= L {
				break
			}
		}

		currentNode := graph.Nodes[current.id]
		for _, neighborID := range currentNode.OutEdges {
			if _, exists := visited[neighborID]; !exists {
				neighborNode := graph.Nodes[neighborID]
				neighborDist := euclideanDistance(neighborNode.Vector, target)

				if resultPool.Len() < L {
					heap.Push(resultPool, &candidate{id: neighborID, dist: neighborDist})
				} else if neighborDist < (*resultPool)[0].dist { // Si el nuevo es mejor que el peor del pool
					heap.Pop(resultPool)                                                  // Eliminar el peor
					heap.Push(resultPool, &candidate{id: neighborID, dist: neighborDist}) // Insertar el nuevo
				}

				heap.Push(candidates, &candidate{id: neighborID, dist: neighborDist})
				visitedNodesForPruning = append(visitedNodesForPruning, neighborNode)
			}
		}
	}

	finalIDs := make([]int, resultPool.Len())
	for i := range finalIDs {
		finalIDs[i] = (*resultPool)[i].id
	}

	return finalIDs, visitedNodesForPruning
}

/*
func BuildVamanaGraph(vectors [][]float64, alpha float64, maxDegree int) *Graph {
	n := len(vectors)
	if n == 0 {
		return nil
	}

	graph := &Graph{
		Nodes:     make([]*Node, n),
		Alpha:     alpha,
		MaxDegree: maxDegree,
	}

	for i := range vectors {
		graph.Nodes[i] = &Node{
			ID:       i,
			Vector:   vectors[i],
			OutEdges: []int{},
		}
	}

	if n == 1 {
		graph.MedoidID = 0
		return graph
	} else {
		graph.MedoidID = computeMedoid(vectors)
	}

	for pass := 0; pass < 2; pass++ {
		currentAlpha := ConstructionAlpha
		if pass == 1 {
			currentAlpha = alpha
		}

		order := rand.Perm(n)
		for _, i := range order {
			node := graph.Nodes[i]

			visitedIDs := GreedySearch(graph, graph.MedoidID, node.Vector, DefaultL)

			candidates := make([]*Node, 0, len(visitedIDs))
			for _, id := range visitedIDs {
				if id != i {
					candidates = append(candidates, graph.Nodes[id])
				}
			}

			if len(candidates) == 0 && i != graph.MedoidID {
				candidates = append(candidates, graph.Nodes[graph.MedoidID])
			}

			newEdges := RobustPrune(node, candidates, currentAlpha, maxDegree, vectors)

			graph.lock.Lock()
			node.OutEdges = newEdges

			for _, neighborID := range newEdges {
				neighbor := graph.Nodes[neighborID]

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

	ensureConnectivity(graph, vectors)

	return graph
}
*/

func ensureConnectivity(graph *Graph, vectors [][]float64) {
	visited := make([]bool, len(graph.Nodes))
	queue := []int{graph.MedoidID}
	visited[graph.MedoidID] = true

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

	for i, isVisited := range visited {
		if !isVisited {
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

func randomEdges(totalNodes, numEdges int) []int {
	edges := make([]int, 0, numEdges)
	for i := 0; i < numEdges; i++ {
		edges = append(edges, rand.Intn(totalNodes))
	}
	return edges
}

func computeMedoid(vectors [][]float64) int {
	type pair struct{ i, j int }
	distChan := make(chan float64, len(vectors)*(len(vectors)-1)/2)

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

func BuildVamanaGraphForShard(shardVectors map[int][]float64, alpha float64, maxDegree int) *Graph {
	n := len(shardVectors)
	if n == 0 {
		return nil
	}

	// Remap original IDs to local 0-based indices for in-memory processing
	localIDToOrigID := make([]int, 0, n)
	origIDToLocalID := make(map[int]int)
	vectors := make([][]float64, 0, n)

	i := 0
	for origID, vec := range shardVectors {
		localIDToOrigID = append(localIDToOrigID, origID)
		origIDToLocalID[origID] = i
		vectors = append(vectors, vec)
		i++
	}

	graph := &Graph{
		Nodes:     make([]*Node, n),
		Alpha:     alpha,
		MaxDegree: maxDegree,
	}

	for i := range vectors {
		graph.Nodes[i] = &Node{ID: i, Vector: vectors[i], OutEdges: make([]int, 0)}
	}

	if n > 1 {
		graph.MedoidID = computeMedoid(vectors) // Medoid is local
		// Initialize with random edges
		for _, node := range graph.Nodes {
			numEdges := max(2, n/2)
			for j := 0; j < numEdges && j < n; j++ { // Simple random init
				randNeighbor := rand.Intn(n)
				if randNeighbor != node.ID {
					node.OutEdges = append(node.OutEdges, randNeighbor)
				}
			}
		}
	} else {
		graph.MedoidID = 0
	}

	for pass := 0; pass < 2; pass++ {
		currentAlpha := ConstructionAlpha
		if pass == 1 {
			currentAlpha = alpha
		}

		order := rand.Perm(n)
		for _, localID := range order {
			node := graph.Nodes[localID]
			_, visitedNodes := GreedySearch(graph, graph.MedoidID, node.Vector, DefaultL)

			// Prune candidates
			newEdgesLocal := RobustPrune(node, visitedNodes, currentAlpha, maxDegree)
			node.OutEdges = newEdgesLocal

			// Add reciprocal edges
			for _, neighborLocalID := range newEdgesLocal {
				neighbor := graph.Nodes[neighborLocalID]
				// Add edge if not present and within degree bounds
				hasEdge := false
				for _, e := range neighbor.OutEdges {
					if e == localID {
						hasEdge = true
						break
					}
				}
				if !hasEdge {
					neighbor.OutEdges = append(neighbor.OutEdges, localID)
					if len(neighbor.OutEdges) > maxDegree {
						// Prune neighbor's edges
						neighborCandidates := make([]*Node, 0, len(neighbor.OutEdges))
						for _, e := range neighbor.OutEdges {
							neighborCandidates = append(neighborCandidates, graph.Nodes[e])
						}
						neighbor.OutEdges = RobustPrune(neighbor, neighborCandidates, currentAlpha, maxDegree)
					}
				}
			}
		}
	}
	ensureConnectivity(graph, vectors)
	finalGraph := &Graph{Nodes: make([]*Node, n), MaxDegree: maxDegree}
	for i, node := range graph.Nodes {
		origID := localIDToOrigID[i]
		origEdges := make([]int, len(node.OutEdges))
		for j, localEdge := range node.OutEdges {
			origEdges[j] = localIDToOrigID[localEdge]
		}
		finalGraph.Nodes[i] = &Node{ID: origID, Vector: node.Vector, OutEdges: origEdges}
	}

	return finalGraph
}
