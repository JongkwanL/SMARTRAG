package collectors

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health/grpc_health_v1"

	"smartrag-monitoring-agent/internal/config"
	"smartrag-monitoring-agent/pkg/types"
)

// HealthCollector performs health checks on configured services
type HealthCollector struct {
	logger   *zap.Logger
	services []config.ServiceConfig
	client   *http.Client
	stats    *HealthStats
	mu       sync.RWMutex
}

// HealthStats tracks health check statistics
type HealthStats struct {
	TotalChecks    int64
	SuccessfulChecks int64
	FailedChecks   int64
	AverageLatency time.Duration
	LastCheckTime  time.Time
}

// NewHealthCollector creates a new health check collector
func NewHealthCollector(logger *zap.Logger, services []config.ServiceConfig) *HealthCollector {
	// Create HTTP client with reasonable defaults
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: false,
			},
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
		},
	}

	return &HealthCollector{
		logger:   logger,
		services: services,
		client:   client,
		stats:    &HealthStats{},
	}
}

// CheckAll performs health checks on all configured services
func (h *HealthCollector) CheckAll(ctx context.Context) ([]types.ServiceHealth, error) {
	start := time.Now()
	defer func() {
		h.mu.Lock()
		h.stats.LastCheckTime = time.Now()
		h.mu.Unlock()
		
		h.logger.Debug("Health checks completed",
			zap.Duration("duration", time.Since(start)),
			zap.Int("service_count", len(h.services)),
		)
	}()

	if len(h.services) == 0 {
		return []types.ServiceHealth{}, nil
	}

	// Perform health checks concurrently
	resultCh := make(chan types.ServiceHealth, len(h.services))
	var wg sync.WaitGroup

	for _, service := range h.services {
		if !service.Enabled {
			continue
		}

		wg.Add(1)
		go func(svc config.ServiceConfig) {
			defer wg.Done()
			
			checkStart := time.Now()
			health := h.checkService(ctx, svc)
			
			// Update statistics
			h.mu.Lock()
			h.stats.TotalChecks++
			if health.Status == types.HealthStatusHealthy {
				h.stats.SuccessfulChecks++
			} else {
				h.stats.FailedChecks++
			}
			
			// Update average latency using exponential moving average
			latency := time.Since(checkStart)
			if h.stats.AverageLatency == 0 {
				h.stats.AverageLatency = latency
			} else {
				alpha := 0.1
				h.stats.AverageLatency = time.Duration(
					float64(h.stats.AverageLatency)*(1-alpha) + float64(latency)*alpha,
				)
			}
			h.mu.Unlock()
			
			resultCh <- health
		}(service)
	}

	// Wait for all checks to complete
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Collect results
	var results []types.ServiceHealth
	for health := range resultCh {
		results = append(results, health)
	}

	return results, nil
}

// checkService performs a health check on a single service
func (h *HealthCollector) checkService(ctx context.Context, service config.ServiceConfig) types.ServiceHealth {
	start := time.Now()
	
	health := types.ServiceHealth{
		ServiceName: service.Name,
		CheckTime:   start,
		Endpoint:    service.Endpoint,
		CheckType:   service.Type,
		Details:     make(map[string]string),
	}

	// Create context with timeout
	checkCtx, cancel := context.WithTimeout(ctx, service.Timeout)
	defer cancel()

	switch service.Type {
	case "http":
		h.checkHTTP(checkCtx, service, &health)
	case "tcp":
		h.checkTCP(checkCtx, service, &health)
	case "grpc":
		h.checkGRPC(checkCtx, service, &health)
	default:
		health.Status = types.HealthStatusUnknown
		health.Error = fmt.Sprintf("Unknown check type: %s", service.Type)
	}

	health.ResponseTime = time.Since(start)
	
	h.logger.Debug("Health check completed",
		zap.String("service", service.Name),
		zap.String("type", service.Type),
		zap.String("status", string(health.Status)),
		zap.Duration("response_time", health.ResponseTime),
		zap.String("error", health.Error),
	)

	return health
}

// checkHTTP performs an HTTP health check
func (h *HealthCollector) checkHTTP(ctx context.Context, service config.ServiceConfig, health *types.ServiceHealth) {
	method := service.Method
	if method == "" {
		method = "GET"
	}

	var body io.Reader
	if service.Body != "" {
		body = strings.NewReader(service.Body)
	}

	req, err := http.NewRequestWithContext(ctx, method, service.Endpoint, body)
	if err != nil {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("Failed to create request: %v", err)
		return
	}

	// Set headers
	for key, value := range service.Headers {
		req.Header.Set(key, value)
	}

	// Set default headers
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", "SmartRAG-Monitoring-Agent/1.0")
	}

	// Configure client for this request
	client := h.client
	if !service.FollowRedirects {
		client = &http.Client{
			Timeout:   service.Timeout,
			Transport: h.client.Transport,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("Request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	// Add response details
	health.Details["status_code"] = fmt.Sprintf("%d", resp.StatusCode)
	health.Details["content_type"] = resp.Header.Get("Content-Type")
	health.Details["content_length"] = resp.Header.Get("Content-Length")

	// Check status code
	expectedStatus := service.ExpectedStatus
	if expectedStatus == 0 {
		expectedStatus = 200
	}

	if resp.StatusCode == expectedStatus {
		health.Status = types.HealthStatusHealthy
		
		// Try to read response body for additional details
		bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1024)) // Limit to 1KB
		if err == nil && len(bodyBytes) > 0 {
			// Try to parse as JSON for structured health info
			var healthResp map[string]interface{}
			if err := json.Unmarshal(bodyBytes, &healthResp); err == nil {
				if status, ok := healthResp["status"]; ok {
					health.Details["response_status"] = fmt.Sprintf("%v", status)
				}
				if version, ok := healthResp["version"]; ok {
					health.Details["version"] = fmt.Sprintf("%v", version)
				}
			} else {
				// Store as plain text if not JSON
				health.Details["response_body"] = string(bodyBytes)
			}
		}
	} else if resp.StatusCode >= 500 {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("Server error: %d", resp.StatusCode)
	} else if resp.StatusCode >= 400 {
		health.Status = types.HealthStatusDegraded
		health.Error = fmt.Sprintf("Client error: %d", resp.StatusCode)
	} else {
		health.Status = types.HealthStatusDegraded
		health.Error = fmt.Sprintf("Unexpected status: %d", resp.StatusCode)
	}
}

// checkTCP performs a TCP connection health check
func (h *HealthCollector) checkTCP(ctx context.Context, service config.ServiceConfig, health *types.ServiceHealth) {
	dialer := &net.Dialer{
		Timeout: service.Timeout,
	}

	conn, err := dialer.DialContext(ctx, "tcp", service.Endpoint)
	if err != nil {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("Connection failed: %v", err)
		return
	}
	
	conn.Close()
	health.Status = types.HealthStatusHealthy
	health.Details["connection"] = "successful"
}

// checkGRPC performs a gRPC health check
func (h *HealthCollector) checkGRPC(ctx context.Context, service config.ServiceConfig, health *types.ServiceHealth) {
	// Configure gRPC connection
	var opts []grpc.DialOption
	if service.SSL {
		opts = append(opts, grpc.WithTransportCredentials(
			grpc.WithTransportCredentials(insecure.NewCredentials()), // This should be replaced with proper TLS config
		))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Set timeout
	dialCtx, cancel := context.WithTimeout(ctx, service.Timeout)
	defer cancel()

	conn, err := grpc.DialContext(dialCtx, service.Endpoint, opts...)
	if err != nil {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("gRPC dial failed: %v", err)
		return
	}
	defer conn.Close()

	// Create health check client
	healthClient := grpc_health_v1.NewHealthClient(conn)

	// Perform health check
	resp, err := healthClient.Check(ctx, &grpc_health_v1.HealthCheckRequest{
		Service: "", // Empty string checks overall health
	})
	if err != nil {
		health.Status = types.HealthStatusUnhealthy
		health.Error = fmt.Sprintf("gRPC health check failed: %v", err)
		return
	}

	// Parse response
	switch resp.Status {
	case grpc_health_v1.HealthCheckResponse_SERVING:
		health.Status = types.HealthStatusHealthy
		health.Details["grpc_status"] = "SERVING"
	case grpc_health_v1.HealthCheckResponse_NOT_SERVING:
		health.Status = types.HealthStatusUnhealthy
		health.Details["grpc_status"] = "NOT_SERVING"
		health.Error = "Service not serving"
	case grpc_health_v1.HealthCheckResponse_UNKNOWN:
		health.Status = types.HealthStatusUnknown
		health.Details["grpc_status"] = "UNKNOWN"
		health.Error = "Service status unknown"
	default:
		health.Status = types.HealthStatusUnknown
		health.Details["grpc_status"] = fmt.Sprintf("UNRECOGNIZED_%d", int(resp.Status))
		health.Error = "Unrecognized service status"
	}
}

// GetStats returns health check statistics
func (h *HealthCollector) GetStats() *HealthStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	return &HealthStats{
		TotalChecks:      h.stats.TotalChecks,
		SuccessfulChecks: h.stats.SuccessfulChecks,
		FailedChecks:     h.stats.FailedChecks,
		AverageLatency:   h.stats.AverageLatency,
		LastCheckTime:    h.stats.LastCheckTime,
	}
}

// UpdateServices updates the list of services to monitor
func (h *HealthCollector) UpdateServices(services []config.ServiceConfig) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	h.services = services
	h.logger.Info("Updated service list",
		zap.Int("service_count", len(services)),
	)
}