package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	"google.golang.org/grpc"

	"smartrag-vector-server/internal/index"
	"smartrag-vector-server/internal/metrics"
	pb "smartrag-vector-server/proto"
)

type VectorServer struct {
	pb.UnimplementedVectorSearchServiceServer
	
	index  index.VectorIndex
	logger *zap.Logger
	
	// Performance monitoring
	searchCount   int64
	totalLatency  time.Duration
	mutex         sync.RWMutex
}

func NewVectorServer(logger *zap.Logger) *VectorServer {
	return &VectorServer{
		index:  index.NewFAISSIndex(384), // Default dimension for sentence-transformers
		logger: logger,
	}
}

// gRPC Search method
func (s *VectorServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start)
		s.mutex.Lock()
		s.searchCount++
		s.totalLatency += latency
		s.mutex.Unlock()
		
		metrics.SearchLatency.Observe(latency.Seconds())
		metrics.SearchCounter.Inc()
	}()

	// Convert request
	queryVector := req.GetQueryVector()
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("query vector is empty")
	}

	topK := int(req.GetTopK())
	if topK <= 0 {
		topK = 10
	}

	// Perform search
	results, err := s.index.Search(queryVector, topK)
	if err != nil {
		s.logger.Error("Search failed", zap.Error(err))
		return nil, err
	}

	// Convert response
	pbResults := make([]*pb.SearchResult, len(results))
	for i, result := range results {
		pbResults[i] = &pb.SearchResult{
			Id:       result.ID,
			Score:    result.Score,
			Metadata: result.Metadata,
		}
	}

	return &pb.SearchResponse{
		Results: pbResults,
		Took:    time.Since(start).Milliseconds(),
	}, nil
}

// Add vectors to index
func (s *VectorServer) Add(ctx context.Context, req *pb.AddRequest) (*pb.AddResponse, error) {
	start := time.Now()
	
	vectors := req.GetVectors()
	ids := req.GetIds()
	metadata := req.GetMetadata()

	if len(vectors) != len(ids) {
		return nil, fmt.Errorf("vectors and ids length mismatch")
	}

	// Convert vectors
	floatVectors := make([][]float32, len(vectors))
	for i, vector := range vectors {
		floatVectors[i] = vector.GetValues()
	}

	// Add to index
	err := s.index.Add(floatVectors, ids, metadata)
	if err != nil {
		return nil, err
	}

	metrics.IndexSize.Set(float64(s.index.Size()))

	return &pb.AddResponse{
		Success: true,
		Took:    time.Since(start).Milliseconds(),
	}, nil
}

// HTTP handlers for REST API
func (s *VectorServer) setupHTTPRoutes() *gin.Engine {
	router := gin.New()
	router.Use(gin.Recovery())

	// Health check
	router.GET("/health", func(c *gin.Context) {
		s.mutex.RLock()
		avgLatency := float64(0)
		if s.searchCount > 0 {
			avgLatency = float64(s.totalLatency.Nanoseconds()) / float64(s.searchCount) / 1e6 // ms
		}
		s.mutex.RUnlock()

		c.JSON(200, gin.H{
			"status":        "healthy",
			"index_size":    s.index.Size(),
			"search_count":  s.searchCount,
			"avg_latency_ms": avgLatency,
			"memory_mb":     runtime.MemStats{}.Alloc / 1024 / 1024,
		})
	})

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(metrics.Handler()))

	// REST API endpoints
	api := router.Group("/api/v1")
	{
		// Search endpoint
		api.POST("/search", func(c *gin.Context) {
			var req struct {
				QueryVector []float32 `json:"query_vector"`
				TopK        int       `json:"top_k"`
			}

			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(400, gin.H{"error": err.Error()})
				return
			}

			// Perform search using internal method
			start := time.Now()
			results, err := s.index.Search(req.QueryVector, req.TopK)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}

			c.JSON(200, gin.H{
				"results": results,
				"took_ms": time.Since(start).Milliseconds(),
			})
		})

		// Add vectors endpoint
		api.POST("/add", func(c *gin.Context) {
			var req struct {
				Vectors  [][]float32       `json:"vectors"`
				IDs      []string          `json:"ids"`
				Metadata []json.RawMessage `json:"metadata"`
			}

			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(400, gin.H{"error": err.Error()})
				return
			}

			start := time.Now()
			err := s.index.Add(req.Vectors, req.IDs, req.Metadata)
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}

			c.JSON(200, gin.H{
				"success": true,
				"took_ms": time.Since(start).Milliseconds(),
			})
		})

		// Index stats
		api.GET("/stats", func(c *gin.Context) {
			s.mutex.RLock()
			stats := gin.H{
				"index_size":     s.index.Size(),
				"search_count":   s.searchCount,
				"total_latency":  s.totalLatency.Milliseconds(),
			}
			if s.searchCount > 0 {
				stats["avg_latency_ms"] = float64(s.totalLatency.Nanoseconds()) / float64(s.searchCount) / 1e6
			}
			s.mutex.RUnlock()

			c.JSON(200, stats)
		})
	}

	return router
}

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Initialize metrics
	metrics.Init()

	// Create vector server
	vectorServer := NewVectorServer(logger)

	// Start gRPC server
	grpcPort := 9090
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", grpcPort))
	if err != nil {
		log.Fatalf("Failed to listen on gRPC port %d: %v", grpcPort, err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterVectorSearchServiceServer(grpcServer, vectorServer)

	go func() {
		logger.Info("Starting gRPC server", zap.Int("port", grpcPort))
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	// Start HTTP server
	httpPort := 8090
	router := vectorServer.setupHTTPRoutes()
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", httpPort),
		Handler: router,
	}

	go func() {
		logger.Info("Starting HTTP server", zap.Int("port", httpPort))
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to serve HTTP: %v", err)
		}
	}()

	logger.Info("SmartRAG Vector Server started",
		zap.Int("grpc_port", grpcPort),
		zap.Int("http_port", httpPort),
	)

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down servers...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		logger.Fatal("HTTP server forced to shutdown", zap.Error(err))
	}

	grpcServer.GracefulStop()
	logger.Info("Servers exited")
}