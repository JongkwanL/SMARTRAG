package vector

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/chewxy/math32"
)

// Vector represents a vector with metadata
type Vector struct {
	ID       string
	Values   []float32
	Metadata map[string]string
}

// SearchResult represents a search result
type SearchResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
}

// Index interface for vector operations
type Index interface {
	Add(vectors []Vector) error
	Search(queryVector []float32, topK int, threshold float32) ([]SearchResult, error)
	Delete(ids []string) error
	GetStats() IndexStats
	Size() int64
	Save(path string) error
	Load(path string) error
}

// IndexStats represents index statistics
type IndexStats struct {
	TotalVectors     int64
	Dimension        int
	IndexSizeMB      float32
	SearchCount      int64
	AvgSearchTimeMs  float32
	TotalSearchTime  time.Duration
}

// HNSWIndex implements HNSW (Hierarchical Navigable Small World) algorithm
type HNSWIndex struct {
	dimension      int
	maxM           int
	maxM0          int
	ml             float64
	efConstruction int
	
	// Node storage
	nodes     map[string]*HNSWNode
	nodesByID map[int]*HNSWNode
	entryNode *HNSWNode
	
	// Statistics
	stats IndexStats
	
	// Thread safety
	mu sync.RWMutex
	
	// Node ID counter
	nextNodeID int
}

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	ID       int
	VectorID string
	Vector   []float32
	Metadata map[string]string
	Level    int
	
	// Connections at each level
	Connections []map[int]bool // level -> set of connected node IDs
}

// SearchCandidate for priority queue
type SearchCandidate struct {
	NodeID   int
	Distance float32
	Index    int // for heap interface
}

// PriorityQueue implements heap.Interface for SearchCandidate
type PriorityQueue []*SearchCandidate

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Distance < pq[j].Distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*SearchCandidate)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.Index = -1
	*pq = old[0 : n-1]
	return item
}

// NewHNSWIndex creates a new HNSW index
func NewHNSWIndex(dimension, maxM, efConstruction int) *HNSWIndex {
	return &HNSWIndex{
		dimension:      dimension,
		maxM:           maxM,
		maxM0:          maxM * 2,
		ml:             1.0 / math.Log(2.0),
		efConstruction: efConstruction,
		nodes:          make(map[string]*HNSWNode),
		nodesByID:      make(map[int]*HNSWNode),
		stats: IndexStats{
			Dimension: dimension,
		},
	}
}

// Add vectors to the index
func (h *HNSWIndex) Add(vectors []Vector) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	for _, vector := range vectors {
		if len(vector.Values) != h.dimension {
			continue // Skip invalid dimensions
		}
		
		node := &HNSWNode{
			ID:       h.nextNodeID,
			VectorID: vector.ID,
			Vector:   vector.Values,
			Metadata: vector.Metadata,
			Level:    h.selectLevel(),
		}
		h.nextNodeID++
		
		// Initialize connections for each level
		node.Connections = make([]map[int]bool, node.Level+1)
		for i := 0; i <= node.Level; i++ {
			node.Connections[i] = make(map[int]bool)
		}
		
		h.nodes[vector.ID] = node
		h.nodesByID[node.ID] = node
		
		// Insert node into the graph
		h.insertNode(node)
		
		h.stats.TotalVectors++
	}
	
	return nil
}

// Search for similar vectors
func (h *HNSWIndex) Search(queryVector []float32, topK int, threshold float32) ([]SearchResult, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		h.mu.Lock()
		h.stats.SearchCount++
		h.stats.TotalSearchTime += duration
		h.stats.AvgSearchTimeMs = float32(h.stats.TotalSearchTime.Nanoseconds()) / float32(h.stats.SearchCount) / 1e6
		h.mu.Unlock()
	}()
	
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	if len(queryVector) != h.dimension {
		return nil, ErrInvalidDimension
	}
	
	if h.entryNode == nil {
		return []SearchResult{}, nil
	}
	
	// Search from top level down to level 1
	candidates := []*SearchCandidate{{NodeID: h.entryNode.ID, Distance: h.distance(queryVector, h.entryNode.Vector)}}
	
	for level := h.entryNode.Level; level > 0; level-- {
		candidates = h.searchLayer(queryVector, candidates, 1, level)
	}
	
	// Search level 0 with ef = max(topK, efConstruction)
	ef := topK
	if ef < h.efConstruction {
		ef = h.efConstruction
	}
	candidates = h.searchLayer(queryVector, candidates, ef, 0)
	
	// Convert to results and apply threshold
	var results []SearchResult
	for i, candidate := range candidates {
		if i >= topK {
			break
		}
		
		node := h.nodesByID[candidate.NodeID]
		if node != nil && candidate.Distance <= threshold {
			results = append(results, SearchResult{
				ID:       node.VectorID,
				Score:    candidate.Distance,
				Metadata: node.Metadata,
			})
		}
	}
	
	return results, nil
}

// Delete vectors from the index
func (h *HNSWIndex) Delete(ids []string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	for _, id := range ids {
		node, exists := h.nodes[id]
		if !exists {
			continue
		}
		
		// Remove connections to this node
		for level := 0; level <= node.Level; level++ {
			for connectedID := range node.Connections[level] {
				if connectedNode, exists := h.nodesByID[connectedID]; exists {
					delete(connectedNode.Connections[level], node.ID)
				}
			}
		}
		
		// Update entry node if necessary
		if h.entryNode != nil && h.entryNode.ID == node.ID {
			h.findNewEntryNode()
		}
		
		delete(h.nodes, id)
		delete(h.nodesByID, node.ID)
		h.stats.TotalVectors--
	}
	
	return nil
}

// GetStats returns index statistics
func (h *HNSWIndex) GetStats() IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	// Estimate index size (rough calculation)
	estimatedSize := float32(len(h.nodes)) * float32(h.dimension) * 4 // 4 bytes per float32
	h.stats.IndexSizeMB = estimatedSize / (1024 * 1024)
	
	return h.stats
}

// Size returns the number of vectors in the index
func (h *HNSWIndex) Size() int64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.stats.TotalVectors
}

// Save index to file (placeholder implementation)
func (h *HNSWIndex) Save(path string) error {
	// TODO: Implement serialization
	return nil
}

// Load index from file (placeholder implementation)
func (h *HNSWIndex) Load(path string) error {
	// TODO: Implement deserialization
	return nil
}

// Helper methods

func (h *HNSWIndex) selectLevel() int {
	level := int(math.Floor(-math.Log(rand.Float64()) * h.ml))
	return level
}

func (h *HNSWIndex) insertNode(node *HNSWNode) {
	if h.entryNode == nil {
		h.entryNode = node
		return
	}
	
	// Find closest nodes at each level
	candidates := []*SearchCandidate{{NodeID: h.entryNode.ID, Distance: h.distance(node.Vector, h.entryNode.Vector)}}
	
	// Search from top level down to node level + 1
	for level := h.entryNode.Level; level > node.Level; level-- {
		candidates = h.searchLayer(node.Vector, candidates, 1, level)
	}
	
	// Connect at each level from node.Level down to 0
	for level := min(node.Level, h.entryNode.Level); level >= 0; level-- {
		candidates = h.searchLayer(node.Vector, candidates, h.efConstruction, level)
		
		maxConnections := h.maxM
		if level == 0 {
			maxConnections = h.maxM0
		}
		
		// Select best candidates and create connections
		selected := h.selectNeighbors(node, candidates, maxConnections, level)
		for _, candidate := range selected {
			h.addConnection(node.ID, candidate.NodeID, level)
		}
	}
	
	// Update entry node if necessary
	if node.Level > h.entryNode.Level {
		h.entryNode = node
	}
}

func (h *HNSWIndex) searchLayer(queryVector []float32, entryPoints []*SearchCandidate, ef int, level int) []*SearchCandidate {
	visited := make(map[int]bool)
	candidates := &PriorityQueue{}
	dynamic := &PriorityQueue{}
	
	// Initialize with entry points
	for _, ep := range entryPoints {
		if !visited[ep.NodeID] {
			visited[ep.NodeID] = true
			heap.Push(candidates, ep)
			heap.Push(dynamic, &SearchCandidate{NodeID: ep.NodeID, Distance: ep.Distance})
		}
	}
	
	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*SearchCandidate)
		
		if current.Distance > dynamic.Peek().(*SearchCandidate).Distance {
			break
		}
		
		node := h.nodesByID[current.NodeID]
		if node == nil || level >= len(node.Connections) {
			continue
		}
		
		// Check all connections at this level
		for connectedID := range node.Connections[level] {
			if !visited[connectedID] {
				visited[connectedID] = true
				connectedNode := h.nodesByID[connectedID]
				if connectedNode != nil {
					distance := h.distance(queryVector, connectedNode.Vector)
					
					if distance < dynamic.Peek().(*SearchCandidate).Distance || dynamic.Len() < ef {
						heap.Push(candidates, &SearchCandidate{NodeID: connectedID, Distance: distance})
						heap.Push(dynamic, &SearchCandidate{NodeID: connectedID, Distance: distance})
						
						if dynamic.Len() > ef {
							heap.Pop(dynamic)
						}
					}
				}
			}
		}
	}
	
	// Return top ef candidates
	result := make([]*SearchCandidate, 0, dynamic.Len())
	for dynamic.Len() > 0 {
		result = append([]*SearchCandidate{heap.Pop(dynamic).(*SearchCandidate)}, result...)
	}
	
	return result
}

func (h *HNSWIndex) selectNeighbors(node *HNSWNode, candidates []*SearchCandidate, maxConnections int, level int) []*SearchCandidate {
	if len(candidates) <= maxConnections {
		return candidates
	}
	
	// Simple strategy: take closest maxConnections candidates
	return candidates[:maxConnections]
}

func (h *HNSWIndex) addConnection(nodeID1, nodeID2, level int) {
	node1 := h.nodesByID[nodeID1]
	node2 := h.nodesByID[nodeID2]
	
	if node1 != nil && node2 != nil && level < len(node1.Connections) && level < len(node2.Connections) {
		node1.Connections[level][nodeID2] = true
		node2.Connections[level][nodeID1] = true
	}
}

func (h *HNSWIndex) findNewEntryNode() {
	var maxLevel int = -1
	for _, node := range h.nodesByID {
		if node.Level > maxLevel {
			maxLevel = node.Level
			h.entryNode = node
		}
	}
}

func (h *HNSWIndex) distance(a, b []float32) float32 {
	// Cosine similarity (converted to distance)
	var dotProduct, normA, normB float32
	
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 1.0
	}
	
	similarity := dotProduct / (math32.Sqrt(normA) * math32.Sqrt(normB))
	return 1.0 - similarity // Convert similarity to distance
}

func (pq PriorityQueue) Peek() *SearchCandidate {
	if len(pq) == 0 {
		return nil
	}
	return pq[0]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Custom errors
var (
	ErrInvalidDimension = fmt.Errorf("invalid vector dimension")
)