package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"smartrag-streaming-proxy/internal/config"
	"smartrag-streaming-proxy/pkg/types"
)

// StreamingProxy handles communication with the backend streaming service
type StreamingProxy struct {
	config     *config.BackendConfig
	logger     *zap.Logger
	httpClient *http.Client
	stats      *ProxyStats
	mu         sync.RWMutex
}

// ProxyStats holds statistics about proxy operations
type ProxyStats struct {
	RequestsTotal     int64
	RequestsSuccess   int64
	RequestsError     int64
	BytesTransferred  int64
	AverageLatency    time.Duration
	ActiveStreams     int
	TotalStreams      int64
}

// NewStreamingProxy creates a new streaming proxy
func NewStreamingProxy(cfg *config.BackendConfig, logger *zap.Logger) *StreamingProxy {
	// Create HTTP client with optimized settings for streaming
	client := &http.Client{
		Timeout: cfg.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        cfg.MaxIdleConns,
			MaxIdleConnsPerHost: cfg.MaxConnsPerHost,
			IdleConnTimeout:     cfg.IdleConnTimeout,
			TLSHandshakeTimeout: cfg.TLSHandshakeTimeout,
			DisableKeepAlives:   false, // Keep connections alive for streaming
			WriteBufferSize:     8192,
			ReadBufferSize:      8192,
		},
	}

	return &StreamingProxy{
		config:     cfg,
		logger:     logger,
		httpClient: client,
		stats:      &ProxyStats{},
	}
}

// StartStream initiates a streaming request to the backend
func (p *StreamingProxy) StartStream(ctx context.Context, request *types.StreamRequest) (<-chan types.StreamResponse, error) {
	start := time.Now()
	
	// Increment active streams
	p.mu.Lock()
	p.stats.ActiveStreams++
	p.stats.TotalStreams++
	p.mu.Unlock()
	
	defer func() {
		p.mu.Lock()
		p.stats.ActiveStreams--
		p.stats.RequestsTotal++
		p.mu.Unlock()
		
		duration := time.Since(start)
		p.updateAverageLatency(duration)
		
		p.logger.Debug("Stream completed",
			zap.Duration("duration", duration),
			zap.String("session_id", request.SessionID),
		)
	}()

	// Create request body
	requestBody, err := json.Marshal(request)
	if err != nil {
		p.incrementErrorCount()
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/stream", p.config.BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		p.incrementErrorCount()
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")
	httpReq.Header.Set("Connection", "keep-alive")

	// Add session ID if present
	if request.SessionID != "" {
		httpReq.Header.Set("X-Session-ID", request.SessionID)
	}

	// Execute request with retry logic
	resp, err := p.executeWithRetry(httpReq)
	if err != nil {
		p.incrementErrorCount()
		return nil, fmt.Errorf("request failed: %w", err)
	}

	// Check response status
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		p.incrementErrorCount()
		return nil, fmt.Errorf("backend returned status: %d", resp.StatusCode)
	}

	// Create response channel
	responseCh := make(chan types.StreamResponse, 100)

	// Start goroutine to handle streaming response
	go p.handleStreamingResponse(ctx, resp, responseCh, request.SessionID)

	p.incrementSuccessCount()
	return responseCh, nil
}

// executeWithRetry executes HTTP request with retry logic
func (p *StreamingProxy) executeWithRetry(req *http.Request) (*http.Response, error) {
	var lastErr error
	
	for attempt := 0; attempt <= p.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-req.Context().Done():
				return nil, req.Context().Err()
			case <-time.After(p.config.RetryDelay * time.Duration(attempt)):
			}
			
			p.logger.Warn("Retrying request",
				zap.Int("attempt", attempt),
				zap.String("url", req.URL.String()),
			)
		}

		resp, err := p.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		// Check if we should retry based on status code
		if resp.StatusCode >= 500 && resp.StatusCode < 600 {
			resp.Body.Close()
			lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", p.config.MaxRetries, lastErr)
}

// handleStreamingResponse processes the streaming response from backend
func (p *StreamingProxy) handleStreamingResponse(ctx context.Context, resp *http.Response, responseCh chan<- types.StreamResponse, sessionID string) {
	defer func() {
		resp.Body.Close()
		close(responseCh)
		
		p.logger.Debug("Streaming response handler finished",
			zap.String("session_id", sessionID),
		)
	}()

	// Create scanner for reading lines
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 1MB max line size

	// Track current event
	var currentEvent SSEEvent

	for {
		select {
		case <-ctx.Done():
			p.logger.Debug("Context cancelled, stopping stream",
				zap.String("session_id", sessionID),
			)
			return
		default:
		}

		if !scanner.Scan() {
			// Check for error
			if err := scanner.Err(); err != nil {
				p.logger.Error("Scanner error",
					zap.Error(err),
					zap.String("session_id", sessionID),
				)
				
				// Send error to client
				errorResp := types.StreamResponse{
					Type: "error",
					Error: &types.ErrorDetails{
						Code:    "stream_error",
						Message: err.Error(),
					},
				}
				
				select {
				case responseCh <- errorResp:
				case <-ctx.Done():
				}
			}
			return
		}

		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines (they separate events)
		if line == "" {
			if currentEvent.Data != "" {
				// Process complete event
				if response, err := p.parseSSEEvent(&currentEvent); err == nil {
					p.updateBytesTransferred(int64(len(currentEvent.Data)))
					
					select {
					case responseCh <- response:
					case <-ctx.Done():
						return
					}
				} else {
					p.logger.Warn("Failed to parse SSE event",
						zap.Error(err),
						zap.String("data", currentEvent.Data),
						zap.String("session_id", sessionID),
					)
				}
				
				// Reset for next event
				currentEvent = SSEEvent{}
			}
			continue
		}

		// Parse SSE field
		if strings.HasPrefix(line, "data: ") {
			data := line[6:] // Remove "data: " prefix
			if currentEvent.Data != "" {
				currentEvent.Data += "\n"
			}
			currentEvent.Data += data
		} else if strings.HasPrefix(line, "event: ") {
			currentEvent.Event = line[7:] // Remove "event: " prefix
		} else if strings.HasPrefix(line, "id: ") {
			currentEvent.ID = line[4:] // Remove "id: " prefix
		} else if strings.HasPrefix(line, "retry: ") {
			// Handle retry field if needed
		}
	}
}

// SSEEvent represents a Server-Sent Event
type SSEEvent struct {
	ID    string
	Event string
	Data  string
}

// parseSSEEvent parses an SSE event into a StreamResponse
func (p *StreamingProxy) parseSSEEvent(event *SSEEvent) (types.StreamResponse, error) {
	var response types.StreamResponse

	// Handle different event types
	switch event.Event {
	case "":
		// Default event type, try to parse as JSON
		if err := json.Unmarshal([]byte(event.Data), &response); err != nil {
			// If JSON parsing fails, treat as plain text chunk
			response = types.StreamResponse{
				Type:    "chunk",
				Content: event.Data,
			}
		}
	case "chunk":
		response.Type = "chunk"
		response.Content = event.Data
	case "metadata":
		response.Type = "metadata"
		response.Metadata = json.RawMessage(event.Data)
	case "error":
		response.Type = "error"
		var errorDetails types.ErrorDetails
		if err := json.Unmarshal([]byte(event.Data), &errorDetails); err == nil {
			response.Error = &errorDetails
		} else {
			response.Error = &types.ErrorDetails{
				Code:    "unknown_error",
				Message: event.Data,
			}
		}
	case "done":
		response.Type = "done"
	default:
		// Try to parse as JSON
		if err := json.Unmarshal([]byte(event.Data), &response); err != nil {
			response = types.StreamResponse{
				Type:    event.Event,
				Content: event.Data,
			}
		}
	}

	return response, nil
}

// HealthCheck performs a health check against the backend
func (p *StreamingProxy) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", p.config.BaseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// GetStats returns proxy statistics
func (p *StreamingProxy) GetStats() *ProxyStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	return &ProxyStats{
		RequestsTotal:     p.stats.RequestsTotal,
		RequestsSuccess:   p.stats.RequestsSuccess,
		RequestsError:     p.stats.RequestsError,
		BytesTransferred:  p.stats.BytesTransferred,
		AverageLatency:    p.stats.AverageLatency,
		ActiveStreams:     p.stats.ActiveStreams,
		TotalStreams:      p.stats.TotalStreams,
	}
}

// incrementSuccessCount increments success counter
func (p *StreamingProxy) incrementSuccessCount() {
	p.mu.Lock()
	p.stats.RequestsSuccess++
	p.mu.Unlock()
}

// incrementErrorCount increments error counter
func (p *StreamingProxy) incrementErrorCount() {
	p.mu.Lock()
	p.stats.RequestsError++
	p.mu.Unlock()
}

// updateBytesTransferred updates bytes transferred counter
func (p *StreamingProxy) updateBytesTransferred(bytes int64) {
	p.mu.Lock()
	p.stats.BytesTransferred += bytes
	p.mu.Unlock()
}

// updateAverageLatency updates average latency
func (p *StreamingProxy) updateAverageLatency(duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.stats.RequestsTotal == 0 {
		p.stats.AverageLatency = duration
	} else {
		// Calculate exponential moving average
		alpha := 0.1 // Smoothing factor
		p.stats.AverageLatency = time.Duration(
			float64(p.stats.AverageLatency)*(1-alpha) + float64(duration)*alpha,
		)
	}
}