package sse

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"smartrag-streaming-proxy/internal/config"
	"smartrag-streaming-proxy/internal/proxy"
	"smartrag-streaming-proxy/pkg/types"
)

// Handler manages Server-Sent Events connections
type Handler struct {
	config      *config.SSEConfig
	logger      *zap.Logger
	proxy       *proxy.StreamingProxy
	connections map[string]*Connection
	mu          sync.RWMutex
	stats       *Stats
}

// Connection represents an SSE client connection
type Connection struct {
	ID           string
	writer       gin.ResponseWriter
	flusher      http.Flusher
	ctx          context.Context
	cancel       context.CancelFunc
	sessionID    string
	clientIP     string
	userAgent    string
	connectedAt  time.Time
	lastActivity time.Time
	eventsSent   int64
	bytesSent    int64
	send         chan *Event
}

// Event represents an SSE event
type Event struct {
	ID    string `json:"id,omitempty"`
	Event string `json:"event,omitempty"`
	Data  string `json:"data"`
	Retry int    `json:"retry,omitempty"`
}

// Stats holds SSE statistics
type Stats struct {
	mu                sync.RWMutex
	ActiveConnections int
	TotalConnections  int64
	EventsSent        int64
	BytesTransferred  int64
	ErrorCount        int64
}

// NewHandler creates a new SSE handler
func NewHandler(cfg *config.SSEConfig, logger *zap.Logger, proxy *proxy.StreamingProxy) *Handler {
	return &Handler{
		config:      cfg,
		logger:      logger,
		proxy:       proxy,
		connections: make(map[string]*Connection),
		stats:       &Stats{},
	}
}

// HandleConnection handles SSE connection requests
func (h *Handler) HandleConnection(c *gin.Context) {
	// Set SSE headers
	h.setSSEHeaders(c.Writer)

	// Check if response writer supports flushing
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Streaming not supported",
		})
		return
	}

	// Create connection context
	ctx, cancel := context.WithCancel(c.Request.Context())
	
	// Create connection
	conn := &Connection{
		ID:           generateConnectionID(),
		writer:       c.Writer,
		flusher:      flusher,
		ctx:          ctx,
		cancel:       cancel,
		sessionID:    c.Query("session_id"),
		clientIP:     c.ClientIP(),
		userAgent:    c.Request.UserAgent(),
		connectedAt:  time.Now(),
		lastActivity: time.Now(),
		send:         make(chan *Event, 256),
	}

	// Register connection
	h.addConnection(conn)
	defer h.removeConnection(conn)

	h.logger.Info("SSE connection established",
		zap.String("connection_id", conn.ID),
		zap.String("client_ip", conn.clientIP),
		zap.String("session_id", conn.sessionID),
	)

	// Send initial event
	conn.sendEvent(&Event{
		Event: "connected",
		Data:  fmt.Sprintf(`{"connection_id":"%s","timestamp":"%s"}`, conn.ID, time.Now().Format(time.RFC3339)),
	})

	// Handle streaming query if provided
	if query := c.Query("query"); query != "" {
		go h.handleStreamingQuery(conn, query, c.Request.URL.Query())
	}

	// Start heartbeat
	heartbeatTicker := time.NewTicker(h.config.HeartbeatInterval)
	defer heartbeatTicker.Stop()

	// Event loop
	for {
		select {
		case event := <-conn.send:
			if err := conn.sendEvent(event); err != nil {
				h.logger.Error("Failed to send SSE event",
					zap.String("connection_id", conn.ID),
					zap.Error(err),
				)
				return
			}

		case <-heartbeatTicker.C:
			// Send heartbeat
			if err := conn.sendEvent(&Event{
				Event: "heartbeat",
				Data:  fmt.Sprintf(`{"timestamp":"%s"}`, time.Now().Format(time.RFC3339)),
			}); err != nil {
				h.logger.Warn("Heartbeat failed, closing connection",
					zap.String("connection_id", conn.ID),
					zap.Error(err),
				)
				return
			}

		case <-ctx.Done():
			h.logger.Info("SSE connection closed",
				zap.String("connection_id", conn.ID),
				zap.Duration("duration", time.Since(conn.connectedAt)),
			)
			return
		}
	}
}

// HandleStreamingQuery handles streaming query requests via HTTP POST
func (h *Handler) HandleStreamingQuery(c *gin.Context) {
	// Parse request
	var request types.StreamRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid request format",
		})
		return
	}

	// Set SSE headers
	h.setSSEHeaders(c.Writer)

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Streaming not supported",
		})
		return
	}

	// Create connection context
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	// Create temporary connection for this request
	conn := &Connection{
		ID:           generateConnectionID(),
		writer:       c.Writer,
		flusher:      flusher,
		ctx:          ctx,
		cancel:       cancel,
		sessionID:    request.SessionID,
		clientIP:     c.ClientIP(),
		userAgent:    c.Request.UserAgent(),
		connectedAt:  time.Now(),
		lastActivity: time.Now(),
		send:         make(chan *Event, 256),
	}

	h.addConnection(conn)
	defer h.removeConnection(conn)

	// Send initial event
	conn.sendEvent(&Event{
		Event: "stream_start",
		Data:  fmt.Sprintf(`{"query":"%s","timestamp":"%s"}`, request.Query, time.Now().Format(time.RFC3339)),
	})

	// Start streaming
	request.Stream = true
	stream, err := h.proxy.StartStream(ctx, &request)
	if err != nil {
		conn.sendEvent(&Event{
			Event: "error",
			Data:  fmt.Sprintf(`{"error":"Failed to start stream: %s"}`, err.Error()),
		})
		return
	}

	// Handle streaming response
	h.handleStreamingResponse(conn, stream)
}

// setSSEHeaders sets the required headers for SSE
func (h *Handler) setSSEHeaders(w gin.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	if h.config.EnableCORS {
		for _, origin := range h.config.AllowedOrigins {
			if origin == "*" {
				w.Header().Set("Access-Control-Allow-Origin", "*")
				break
			}
		}
	}
}

// addConnection adds a connection to the handler
func (h *Handler) addConnection(conn *Connection) {
	h.mu.Lock()
	h.connections[conn.ID] = conn
	h.stats.mu.Lock()
	h.stats.ActiveConnections++
	h.stats.TotalConnections++
	h.stats.mu.Unlock()
	h.mu.Unlock()

	h.logger.Debug("SSE connection added",
		zap.String("connection_id", conn.ID),
		zap.Int("active_connections", len(h.connections)),
	)
}

// removeConnection removes a connection from the handler
func (h *Handler) removeConnection(conn *Connection) {
	h.mu.Lock()
	if _, exists := h.connections[conn.ID]; exists {
		delete(h.connections, conn.ID)
		conn.cancel()
		h.stats.mu.Lock()
		h.stats.ActiveConnections--
		h.stats.mu.Unlock()
	}
	h.mu.Unlock()

	h.logger.Debug("SSE connection removed",
		zap.String("connection_id", conn.ID),
		zap.Int("active_connections", len(h.connections)),
	)
}

// handleStreamingQuery handles streaming query for a connection
func (h *Handler) handleStreamingQuery(conn *Connection, query string, params map[string][]string) {
	// Build request
	request := &types.StreamRequest{
		Query:     query,
		SessionID: conn.sessionID,
		Stream:    true,
		Parameters: make(map[string]interface{}),
	}

	// Add URL parameters
	for key, values := range params {
		if len(values) > 0 {
			request.Parameters[key] = values[0]
		}
	}

	// Start streaming
	stream, err := h.proxy.StartStream(conn.ctx, request)
	if err != nil {
		conn.send <- &Event{
			Event: "error",
			Data:  fmt.Sprintf(`{"error":"Failed to start stream: %s"}`, err.Error()),
		}
		return
	}

	// Handle streaming response
	h.handleStreamingResponse(conn, stream)
}

// handleStreamingResponse handles streaming responses from backend
func (h *Handler) handleStreamingResponse(conn *Connection, stream <-chan types.StreamResponse) {
	for {
		select {
		case response, ok := <-stream:
			if !ok {
				// Stream closed
				conn.send <- &Event{
					Event: "stream_end",
					Data:  fmt.Sprintf(`{"timestamp":"%s"}`, time.Now().Format(time.RFC3339)),
				}
				return
			}

			// Convert response to JSON
			data, err := json.Marshal(response)
			if err != nil {
				h.logger.Error("Failed to marshal stream response", zap.Error(err))
				continue
			}

			// Send event
			event := &Event{
				Event: response.Type,
				Data:  string(data),
			}

			select {
			case conn.send <- event:
				h.updateEventsSent()
			default:
				// Channel full, connection will be closed
				return
			}

		case <-conn.ctx.Done():
			return
		}
	}
}

// sendEvent sends an SSE event to the client
func (c *Connection) sendEvent(event *Event) error {
	var data string

	// Format SSE event
	if event.ID != "" {
		data += fmt.Sprintf("id: %s\n", event.ID)
	}
	if event.Event != "" {
		data += fmt.Sprintf("event: %s\n", event.Event)
	}
	if event.Retry > 0 {
		data += fmt.Sprintf("retry: %d\n", event.Retry)
	}

	// Split data by lines and prefix each with "data: "
	if event.Data != "" {
		lines := splitLines(event.Data)
		for _, line := range lines {
			data += fmt.Sprintf("data: %s\n", line)
		}
	}

	data += "\n" // Empty line to end the event

	// Write to response
	if _, err := c.writer.Write([]byte(data)); err != nil {
		return err
	}

	// Flush the response
	c.flusher.Flush()

	// Update stats
	c.eventsSent++
	c.bytesSent += int64(len(data))
	c.lastActivity = time.Now()

	return nil
}

// splitLines splits text into lines for SSE format
func splitLines(text string) []string {
	lines := make([]string, 0)
	current := ""
	
	for _, char := range text {
		if char == '\n' || char == '\r' {
			if current != "" {
				lines = append(lines, current)
				current = ""
			}
		} else {
			current += string(char)
		}
	}
	
	if current != "" {
		lines = append(lines, current)
	}
	
	if len(lines) == 0 {
		lines = append(lines, text)
	}
	
	return lines
}

// generateConnectionID generates a unique connection ID
func generateConnectionID() string {
	return fmt.Sprintf("sse_%d_%d", time.Now().UnixNano(), time.Now().Nanosecond())
}

// updateEventsSent updates the events sent counter
func (h *Handler) updateEventsSent() {
	h.stats.mu.Lock()
	h.stats.EventsSent++
	h.stats.mu.Unlock()
}

// GetStats returns current SSE statistics
func (h *Handler) GetStats() *Stats {
	h.stats.mu.RLock()
	defer h.stats.mu.RUnlock()
	
	return &Stats{
		ActiveConnections: h.stats.ActiveConnections,
		TotalConnections:  h.stats.TotalConnections,
		EventsSent:        h.stats.EventsSent,
		BytesTransferred:  h.stats.BytesTransferred,
		ErrorCount:        h.stats.ErrorCount,
	}
}

// BroadcastEvent broadcasts an event to all connected clients
func (h *Handler) BroadcastEvent(event *Event) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	for _, conn := range h.connections {
		select {
		case conn.send <- event:
		default:
			// Channel full, skip this connection
		}
	}
}