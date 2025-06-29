package grpc

import (
	"context"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "smartrag-vector-search/pkg/proto"
	"smartrag-vector-search/internal/vector"
	"smartrag-vector-search/internal/metrics"
)

// VectorSearchServer implements the gRPC service
type VectorSearchServer struct {
	pb.UnimplementedVectorSearchServiceServer
	index   vector.Index
	logger  *zap.Logger
	metrics *metrics.Metrics
}

// NewVectorSearchServer creates a new gRPC server instance
func NewVectorSearchServer(index vector.Index, logger *zap.Logger, metrics *metrics.Metrics) *VectorSearchServer {
	return &VectorSearchServer{
		index:   index,
		logger:  logger,
		metrics: metrics,
	}
}

// Search performs vector similarity search
func (s *VectorSearchServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.metrics.RecordSearchLatency(duration)
		s.logger.Debug("Search completed",
			zap.Duration("duration", duration),
			zap.Int32("top_k", req.TopK),
			zap.Float32("threshold", req.Threshold),
		)
	}()

	// Validate request
	if len(req.QueryVector) == 0 {
		s.metrics.IncSearchErrors("empty_query_vector")
		return nil, status.Error(codes.InvalidArgument, "query vector cannot be empty")
	}

	if req.TopK <= 0 {
		req.TopK = 10 // Default value
	}

	if req.Threshold <= 0 {
		req.Threshold = 1.0 // Default to no threshold
	}

	// Perform search
	results, err := s.index.Search(req.QueryVector, int(req.TopK), req.Threshold)
	if err != nil {
		s.metrics.IncSearchErrors("search_failed")
		s.logger.Error("Search failed", zap.Error(err))
		return nil, status.Error(codes.Internal, "search operation failed")
	}

	// Convert results to protobuf format
	pbResults := make([]*pb.SearchResult, len(results))
	for i, result := range results {
		pbResult := &pb.SearchResult{
			Id:    result.ID,
			Score: result.Score,
		}

		// Include metadata if requested
		if req.IncludeMetadata && result.Metadata != nil {
			pbResult.Metadata = result.Metadata
		}

		pbResults[i] = pbResult
	}

	s.metrics.IncSearchRequests()
	s.metrics.RecordSearchResults(len(results))

	return &pb.SearchResponse{
		Results:        pbResults,
		TotalSearched:  s.index.Size(),
		SearchTimeMs:   float32(time.Since(start).Nanoseconds()) / 1e6,
	}, nil
}

// AddVectors adds vectors to the index
func (s *VectorSearchServer) AddVectors(ctx context.Context, req *pb.AddVectorsRequest) (*pb.AddVectorsResponse, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.metrics.RecordAddLatency(duration)
		s.logger.Debug("AddVectors completed",
			zap.Duration("duration", duration),
			zap.Int("vector_count", len(req.Vectors)),
		)
	}()

	if len(req.Vectors) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vectors list cannot be empty")
	}

	// Convert protobuf vectors to internal format
	vectors := make([]vector.Vector, len(req.Vectors))
	var failedIDs []string

	for i, pbVector := range req.Vectors {
		if len(pbVector.Values) == 0 {
			failedIDs = append(failedIDs, pbVector.Id)
			continue
		}

		vectors[i] = vector.Vector{
			ID:       pbVector.Id,
			Values:   pbVector.Values,
			Metadata: pbVector.Metadata,
		}
	}

	// Filter out failed vectors
	validVectors := make([]vector.Vector, 0, len(vectors))
	for _, v := range vectors {
		if v.ID != "" && len(v.Values) > 0 {
			validVectors = append(validVectors, v)
		}
	}

	// Add vectors to index
	if len(validVectors) > 0 {
		err := s.index.Add(validVectors)
		if err != nil {
			s.metrics.IncAddErrors("add_failed")
			s.logger.Error("Failed to add vectors", zap.Error(err))
			return nil, status.Error(codes.Internal, "failed to add vectors to index")
		}
	}

	s.metrics.IncAddRequests()
	s.metrics.RecordVectorsAdded(len(validVectors))

	return &pb.AddVectorsResponse{
		AddedCount: int32(len(validVectors)),
		FailedIds:  failedIDs,
	}, nil
}

// DeleteVectors removes vectors from the index
func (s *VectorSearchServer) DeleteVectors(ctx context.Context, req *pb.DeleteVectorsRequest) (*pb.DeleteVectorsResponse, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.metrics.RecordDeleteLatency(duration)
		s.logger.Debug("DeleteVectors completed",
			zap.Duration("duration", duration),
			zap.Int("id_count", len(req.Ids)),
		)
	}()

	if len(req.Ids) == 0 {
		return nil, status.Error(codes.InvalidArgument, "ids list cannot be empty")
	}

	// Track which IDs were not found
	initialSize := s.index.Size()

	err := s.index.Delete(req.Ids)
	if err != nil {
		s.metrics.IncDeleteErrors("delete_failed")
		s.logger.Error("Failed to delete vectors", zap.Error(err))
		return nil, status.Error(codes.Internal, "failed to delete vectors from index")
	}

	finalSize := s.index.Size()
	deletedCount := int(initialSize - finalSize)

	// Determine which IDs were not found (simple approximation)
	var notFoundIDs []string
	if deletedCount < len(req.Ids) {
		// In a real implementation, you'd track this more precisely
		remainingCount := len(req.Ids) - deletedCount
		notFoundIDs = req.Ids[len(req.Ids)-remainingCount:]
	}

	s.metrics.IncDeleteRequests()
	s.metrics.RecordVectorsDeleted(deletedCount)

	return &pb.DeleteVectorsResponse{
		DeletedCount:  int32(deletedCount),
		NotFoundIds:   notFoundIDs,
	}, nil
}

// GetStats returns index statistics
func (s *VectorSearchServer) GetStats(ctx context.Context, req *pb.GetStatsRequest) (*pb.GetStatsResponse, error) {
	stats := s.index.GetStats()

	return &pb.GetStatsResponse{
		TotalVectors:    stats.TotalVectors,
		Dimension:       int32(stats.Dimension),
		IndexSizeMb:     stats.IndexSizeMB,
		SearchCount:     stats.SearchCount,
		AvgSearchTimeMs: stats.AvgSearchTimeMs,
	}, nil
}

// HealthCheck performs a health check
func (s *VectorSearchServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	// Perform basic health checks
	indexSize := s.index.Size()
	
	// Check if index is responsive
	if indexSize < 0 {
		return &pb.HealthCheckResponse{
			Status:    "unhealthy",
			Timestamp: time.Now().Unix(),
		}, nil
	}

	return &pb.HealthCheckResponse{
		Status:    "healthy",
		Timestamp: time.Now().Unix(),
	}, nil
}