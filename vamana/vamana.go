package vamana

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
)

const (
	DefaultAlpha      = 1.2
	DefaultL          = 100
	DefaultMaxDegree  = 64
	ConstructionAlpha = 1.0
	DefaultBeamWidth  = 4
)

var totalPruneC = 0

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
	dCache   map[int]float64
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

type minHeap []*candidate

func (mh minHeap) Len() int           { return len(mh) }
func (mh minHeap) Less(i, j int) bool { return mh[i].dist < mh[j].dist }
func (mh minHeap) Swap(i, j int)      { mh[i], mh[j] = mh[j], mh[i] }

func (mh *minHeap) Push(x interface{}) {
	item := x.(*candidate)
	*mh = append(*mh, item)
}

func (mh *minHeap) Pop() interface{} {
	old := *mh
	n := len(old)
	item := old[n-1]
	*mh = old[0 : n-1]
	return item
}

// src -> dst distance
func distanceCacheWithNode(src *Node, dst *Node) float64 {
	if dist, ok := dst.dCache[src.ID]; ok {
		return dist
	} else {
		mdist := euclideanDistanceUnsafe(src.Vector, dst.Vector)
		dst.dCache[src.ID] = mdist
		return mdist
	}
}

func RobustPrune(p *Node, candidates []*Node, alpha float64, maxDegree int) []int {
	totalPruneC++
	if len(candidates) == 0 {
		return nil
	}
	//distanceCache := make(map[int]float64, len(candidates))
	sortedCandidates := make([]*Node, len(candidates))
	nodeCache := make(map[int]*Node, len(candidates))
	for _, candidate := range candidates {

		distanceCacheWithNode(p, candidate)
		nodeCache[candidate.ID] = candidate

	}

	// Create a copy to avoid modifying the original slice of candidates

	copy(sortedCandidates, candidates)

	sort.Slice(sortedCandidates, func(i, j int) bool {
		return sortedCandidates[i].dCache[p.ID] <
			sortedCandidates[j].dCache[p.ID]
	})

	pruned := make([]int, 0, maxDegree)

	for _, candidate := range sortedCandidates {
		if len(pruned) >= maxDegree {
			break
		}

		keep := true
		distToCandidate := candidate.dCache[p.ID]

		for _, existingID := range pruned {
			// Need the full node to get the vector
			existingNode, ok := nodeCache[existingID]
			if !ok {
				continue
			}

			// This part is tricky without a map from ID to Node.
			// Let's assume we can find it for now. A map would be more efficient.
			if existingNode != nil {
				//distBetweenSelected := euclideanDistanceUnsafe(existingNode.Vector, candidate.Vector)

				distBetweenSelected := distanceCacheWithNode(existingNode, candidate)

				if alpha*distBetweenSelected < distToCandidate {
					keep = false
					break
				}
			}
		}

		if keep {
			if candidate.ID != p.ID {
				pruned = append(pruned, candidate.ID)
			}
		}
	}

	return pruned
}

func GreedySearch(graph *Graph, startID int, target []float64, L int) ([]int, []*Node) {
	visited := make(map[int]struct{})
	visitedMap := make(map[int]*Node, L*2)

	candidates := &minHeap{}
	heap.Init(candidates)

	// Add start node to candidates
	dist := euclideanDistanceUnsafe(graph.Nodes[startID].Vector, target)
	heap.Push(candidates, &candidate{id: startID, dist: dist})

	resultPool := &priorityQueue{}
	heap.Init(resultPool)
	heap.Push(resultPool, &candidate{id: startID, dist: dist})

	//visitedNodesForPruning := []*Node{graph.Nodes[startID]}

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*candidate)

		if _, exists := visited[current.id]; exists {
			continue
		}

		visited[current.id] = struct{}{}

		if resultPool.Len() >= L && current.dist > (*resultPool)[0].dist {
			break
		}

		currentNode := graph.Nodes[current.id]
		for _, neighborID := range currentNode.OutEdges {
			if _, exists := visited[neighborID]; !exists {

				neighborNode := graph.Nodes[neighborID]

				neighborDist := euclideanDistanceUnsafe(neighborNode.Vector, target)

				if resultPool.Len() < L {
					heap.Push(resultPool, &candidate{id: neighborID, dist: neighborDist})
				} else if neighborDist < (*resultPool)[0].dist { // Si el nuevo es mejor que el peor del pool
					heap.Pop(resultPool)                                                  // Eliminar el peor
					heap.Push(resultPool, &candidate{id: neighborID, dist: neighborDist}) // Insertar el nuevo
				}

				heap.Push(candidates, &candidate{id: neighborID, dist: neighborDist})

				//visitedNodesForPruning = append(visitedNodesForPruning, neighborNode)
				if _, ok := visitedMap[neighborID]; !ok {
					visitedMap[neighborID] = neighborNode
				}
			}
		}
	}

	finalIDs := make([]int, resultPool.Len())
	for i := range finalIDs {
		finalIDs[i] = (*resultPool)[i].id
	}

	visitedNodesForPruning := make([]*Node, 0, len(visitedMap))

	for _, n := range visitedMap {
		visitedNodesForPruning = append(visitedNodesForPruning, n)
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
					dist := euclideanDistanceUnsafe(vectors[i], vectors[j])
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
	//O(N^2)
	var wg sync.WaitGroup
	for i := 0; i < len(vectors); i++ {
		for j := i + 1; j < len(vectors); j++ {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				dist := euclideanDistanceUnsafe(vectors[i], vectors[j])
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
				total += euclideanDistanceUnsafe(vectors[i], vectors[j])
			}
		}
		if total < minTotal {
			minTotal = total
			medoid = i
		}
	}
	return medoid
}

func BuildVamanaGraphForShard(vectors [][]float64, alpha float64, maxDegree int) *Graph {
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
		graph.Nodes[i] = &Node{ID: i, Vector: vectors[i], OutEdges: make([]int, 0, maxDegree*2), dCache: map[int]float64{}}
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

	return graph
}
func (g *Graph) GenerateDotGraphOptimized() string {
	g.lock.RLock()
	defer g.lock.RUnlock()

	var sb strings.Builder
	// Use 'neato' layout engine for absolute positioning based on 'pos' attribute
	// or 'fdp' for force-directed layouts that might consider distances more organically
	// 'neato' is good when you want to explicitly control positions.
	sb.WriteString("digraph G {\n")
	sb.WriteString("  overlap=false;\n")                                                                     // Prevent node overlap
	sb.WriteString("  splines=true;\n")                                                                      // Draw smooth lines
	sb.WriteString("  node [shape=circle, style=filled, fillcolor=lightblue, fixedsize=true, width=0.8];\n") // Styling for nodes

	// Iterate over nodes to define them and their attributes, including position
	for _, node := range g.Nodes {
		if node == nil {
			continue
		}

		// Prepare label for the node
		label := fmt.Sprintf("ID: %d\\nVec: %v", node.ID, node.Vector)

		// Try to use the first two dimensions of the vector as X, Y coordinates
		posAttribute := ""
		if len(node.Vector) >= 2 {
			// Convert to string in Graphviz 'x,y!' format for absolute positioning
			// Multiplied by a factor (e.g., 50) for better spacing in the graph
			x := node.Vector[0] * 50
			y := node.Vector[1] * 50
			posAttribute = fmt.Sprintf(" pos=\"%f,%f!\"", x, y) // '!' makes position fixed
		}

		// Define node with its ID, label, and position
		sb.WriteString(fmt.Sprintf("  %d [label=\"%s\"%s];\n", node.ID, label, posAttribute))
	}

	// Iterate over nodes to define edges (connections)
	for _, node := range g.Nodes {
		if node == nil {
			continue
		}
		for _, edgeID := range node.OutEdges {
			// Add a directed edge from the current node to the edgeID node
			sb.WriteString(fmt.Sprintf("  %d -> %d;\n", node.ID, edgeID))
		}
	}

	// Highlight the medoid node
	// Find the medoid node to ensure its vector is included in its label if its ID isn't directly the index
	var medoidNode *Node
	for _, node := range g.Nodes {
		if node != nil && node.ID == g.MedoidID {
			medoidNode = node
			break
		}
	}

	if medoidNode != nil {
		medoidLabel := fmt.Sprintf("Medoid\\nID: %d\\nVec: %v", medoidNode.ID, medoidNode.Vector)
		posAttribute := ""
		if len(medoidNode.Vector) >= 2 {
			x := medoidNode.Vector[0] * 50
			y := medoidNode.Vector[1] * 50
			posAttribute = fmt.Sprintf(" pos=\"%f,%f!\"", x, y)
		}
		// Make medoid bigger and red
		sb.WriteString(fmt.Sprintf("  %d [fillcolor=red, width=1.0, label=\"%s\"%s];\n", medoidNode.ID, medoidLabel, posAttribute))
	}

	sb.WriteString("}\n")
	return sb.String()
}
