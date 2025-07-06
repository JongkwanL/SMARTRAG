package types

import (
	"encoding/json"
	"time"
)

// StreamMessage represents a message in the streaming response
type StreamMessage struct {
	ID        string                 `json:"id,omitempty"`
	Event     string                 `json:"event,omitempty"`
	Data      json.RawMessage        `json:"data,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// StreamRequest represents a streaming request to the backend
type StreamRequest struct {
	Query     string                 `json:"query"`
	SessionID string                 `json:"session_id,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Stream    bool                   `json:"stream"`
}

// StreamResponse represents the structure of streaming responses from backend
type StreamResponse struct {
	Type      string          `json:"type"`      // "chunk", "metadata", "error", "done"
	Content   string          `json:"content,omitempty"`
	Delta     string          `json:"delta,omitempty"`
	TokenUsage *TokenUsage    `json:"token_usage,omitempty"`
	Sources   []Source        `json:"sources,omitempty"`
	Error     *ErrorDetails   `json:"error,omitempty"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
}

// TokenUsage represents token usage statistics
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Source represents a source document used in RAG
type Source struct {
	ID       string                 `json:"id"`
	Title    string                 `json:"title,omitempty"`
	Content  string                 `json:"content"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ErrorDetails represents error information
type ErrorDetails struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// ConnectionInfo represents information about a client connection
type ConnectionInfo struct {
	ID         string    `json:"id"`
	ClientIP   string    `json:"client_ip"`
	UserAgent  string    `json:"user_agent"`
	Protocol   string    `json:"protocol"` // "websocket" or "sse"
	ConnectedAt time.Time `json:"connected_at"`
	SessionID  string    `json:"session_id,omitempty"`
}

// ClientMessage represents a message from client to server
type ClientMessage struct {
	Type    string          `json:"type"`    // "query", "ping", "close"
	Payload json.RawMessage `json:"payload"`
}

// ServerMessage represents a message from server to client
type ServerMessage struct {
	Type      string          `json:"type"`      // "chunk", "error", "pong", "close"
	ID        string          `json:"id,omitempty"`
	Timestamp time.Time       `json:"timestamp"`
	Payload   json.RawMessage `json:"payload"`
}

// HealthStatus represents the health status of the service
type HealthStatus struct {
	Status      string            `json:"status"`      // "healthy", "unhealthy", "degraded"
	Timestamp   time.Time         `json:"timestamp"`
	Version     string            `json:"version"`
	Uptime      time.Duration     `json:"uptime"`
	Connections int               `json:"connections"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
}

// ProxyStats represents statistics about the proxy
type ProxyStats struct {
	ActiveConnections int                `json:"active_connections"`
	TotalConnections  int64              `json:"total_connections"`
	MessagesProcessed int64              `json:"messages_processed"`
	BytesTransferred  int64              `json:"bytes_transferred"`
	ErrorCount        int64              `json:"error_count"`
	Uptime           time.Duration       `json:"uptime"`
	MemoryUsage      int64              `json:"memory_usage"`
	ConnectionsByType map[string]int     `json:"connections_by_type"`
}

// RateLimitInfo represents rate limiting information
type RateLimitInfo struct {
	Allowed   bool  `json:"allowed"`
	Remaining int   `json:"remaining"`
	Reset     int64 `json:"reset"`
	Limit     int   `json:"limit"`
}

// WebSocketUpgradeRequest represents a WebSocket upgrade request
type WebSocketUpgradeRequest struct {
	Headers     map[string]string `json:"headers"`
	Origin      string           `json:"origin"`
	Protocol    string           `json:"protocol,omitempty"`
	Extensions  []string         `json:"extensions,omitempty"`
	SessionID   string           `json:"session_id,omitempty"`
}

// SSEConnection represents an SSE connection state
type SSEConnection struct {
	ID           string    `json:"id"`
	Connected    time.Time `json:"connected"`
	LastActivity time.Time `json:"last_activity"`
	EventsSent   int64     `json:"events_sent"`
	BytesSent    int64     `json:"bytes_sent"`
}

// BackendStatus represents the status of backend connection
type BackendStatus struct {
	URL         string        `json:"url"`
	Status      string        `json:"status"` // "connected", "disconnected", "error"
	LastCheck   time.Time     `json:"last_check"`
	ResponseTime time.Duration `json:"response_time"`
	ErrorCount  int64         `json:"error_count"`
}