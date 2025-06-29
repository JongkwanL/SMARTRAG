package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"smartrag-vector-search/internal/config"
	grpcServer "smartrag-vector-search/internal/grpc"
	"smartrag-vector-search/internal/metrics"
	"smartrag-vector-search/internal/vector"
	pb "smartrag-vector-search/pkg/proto"
)

func main() {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize logger: %v", err))
	}
	defer logger.Sync()

	// Load configuration
	cfg := config.Load()
	logger.Info("Configuration loaded", 
		zap.String("host", cfg.Server.Host),
		zap.Int("port", cfg.Server.Port),
		zap.Int("dimension", cfg.Vector.Dimension),
		zap.String("index_type", cfg.Vector.IndexType),
	)

	// Initialize metrics
	metricsCollector := metrics.NewMetrics()
	logger.Info("Metrics initialized")

	// Start metrics server if enabled
	if cfg.Metrics.Enabled {
		go func() {
			addr := fmt.Sprintf(":%d", cfg.Metrics.Port)
			logger.Info("Starting metrics server", zap.String("addr", addr))
			if err := metricsCollector.StartMetricsServer(addr); err != nil {
				logger.Error("Metrics server failed", zap.Error(err))
			}
		}()
	}

	// Initialize vector index
	var index vector.Index
	switch cfg.Vector.IndexType {
	case "hnsw":
		index = vector.NewHNSWIndex(
			cfg.Vector.Dimension,
			cfg.Vector.M,
			cfg.Vector.EfConstruction,
		)
		logger.Info("HNSW index initialized",
			zap.Int("dimension", cfg.Vector.Dimension),
			zap.Int("M", cfg.Vector.M),
			zap.Int("ef_construction", cfg.Vector.EfConstruction),
		)
	default:
		logger.Fatal("Unsupported index type", zap.String("type", cfg.Vector.IndexType))
	}

	// Load existing index if available
	if cfg.Vector.IndexPath != "" {
		if err := index.Load(cfg.Vector.IndexPath); err != nil {
			logger.Warn("Failed to load existing index", 
				zap.Error(err),
				zap.String("path", cfg.Vector.IndexPath),
			)
		} else {
			logger.Info("Loaded existing index",
				zap.String("path", cfg.Vector.IndexPath),
				zap.Int64("vectors", index.Size()),
			)
		}
	}

	// Start system metrics collection
	go collectSystemMetrics(metricsCollector, logger)

	// Start index metrics collection
	go collectIndexMetrics(index, metricsCollector, logger)

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor(logger)),
		grpc.MaxRecvMsgSize(16*1024*1024), // 16MB
		grpc.MaxSendMsgSize(16*1024*1024), // 16MB
	)

	// Register vector search service
	vectorService := grpcServer.NewVectorSearchServer(index, logger, metricsCollector)
	pb.RegisterVectorSearchServiceServer(grpcServer, vectorService)

	// Enable reflection for debugging
	reflection.Register(grpcServer)

	// Create listener
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("Failed to create listener", 
			zap.Error(err),
			zap.String("addr", addr),
		)
	}

	// Start server in goroutine
	go func() {
		logger.Info("Starting vector search server", zap.String("addr", addr))
		if err := grpcServer.Serve(listener); err != nil {
			logger.Error("gRPC server failed", zap.Error(err))
		}
	}()

	// Wait for termination signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigChan

	logger.Info("Received shutdown signal", zap.String("signal", sig.String()))

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Save index if path is configured
	if cfg.Vector.IndexPath != "" {
		logger.Info("Saving index", zap.String("path", cfg.Vector.IndexPath))
		if err := index.Save(cfg.Vector.IndexPath); err != nil {
			logger.Error("Failed to save index", zap.Error(err))
		} else {
			logger.Info("Index saved successfully")
		}
	}

	// Stop gRPC server
	stopped := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		logger.Info("Server stopped gracefully")
	case <-shutdownCtx.Done():
		logger.Warn("Shutdown timeout exceeded, forcing stop")
		grpcServer.Stop()
	}

	logger.Info("Vector search server shut down")
}

// loggingInterceptor logs gRPC requests
func loggingInterceptor(logger *zap.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		
		if err != nil {
			logger.Error("gRPC request failed",
				zap.String("method", info.FullMethod),
				zap.Duration("duration", duration),
				zap.Error(err),
			)
		} else {
			logger.Info("gRPC request completed",
				zap.String("method", info.FullMethod),
				zap.Duration("duration", duration),
			)
		}
		
		return resp, err
	}
}

// collectSystemMetrics periodically collects system metrics
func collectSystemMetrics(metrics *metrics.Metrics, logger *zap.Logger) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Get memory stats
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)
		
		// Update memory usage metric
		metrics.UpdateMemoryUsage(memStats.Alloc)
		
		// CPU usage would require additional libraries like shirou/gopsutil
		// For now, we'll skip CPU metrics to keep dependencies minimal
		
		logger.Debug("System metrics collected",
			zap.Uint64("memory_alloc", memStats.Alloc),
			zap.Uint64("memory_sys", memStats.Sys),
		)
	}
}

// collectIndexMetrics periodically collects index metrics
func collectIndexMetrics(index vector.Index, metrics *metrics.Metrics, logger *zap.Logger) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		stats := index.GetStats()
		
		metrics.UpdateTotalVectors(stats.TotalVectors)
		metrics.UpdateIndexSize(float64(stats.IndexSizeMB))
		
		logger.Debug("Index metrics collected",
			zap.Int64("total_vectors", stats.TotalVectors),
			zap.Float32("index_size_mb", stats.IndexSizeMB),
			zap.Int64("search_count", stats.SearchCount),
		)
	}
}