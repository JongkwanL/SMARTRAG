package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Backend   BackendConfig   `yaml:"backend"`
	Streaming StreamingConfig `yaml:"streaming"`
	Metrics   MetricsConfig   `yaml:"metrics"`
	Security  SecurityConfig  `yaml:"security"`
}

type ServerConfig struct {
	Host            string        `yaml:"host"`
	Port            int           `yaml:"port"`
	ReadTimeout     time.Duration `yaml:"read_timeout"`
	WriteTimeout    time.Duration `yaml:"write_timeout"`
	IdleTimeout     time.Duration `yaml:"idle_timeout"`
	MaxConnections  int           `yaml:"max_connections"`
	EnableProfiling bool          `yaml:"enable_profiling"`
}

type BackendConfig struct {
	BaseURL             string        `yaml:"base_url"`
	Timeout             time.Duration `yaml:"timeout"`
	MaxIdleConns        int           `yaml:"max_idle_conns"`
	MaxConnsPerHost     int           `yaml:"max_conns_per_host"`
	IdleConnTimeout     time.Duration `yaml:"idle_conn_timeout"`
	TLSHandshakeTimeout time.Duration `yaml:"tls_handshake_timeout"`
	KeepAlive           time.Duration `yaml:"keep_alive"`
	MaxRetries          int           `yaml:"max_retries"`
	RetryDelay          time.Duration `yaml:"retry_delay"`
}

type StreamingConfig struct {
	// WebSocket settings
	WebSocket WebSocketConfig `yaml:"websocket"`
	
	// Server-Sent Events settings
	SSE SSEConfig `yaml:"sse"`
	
	// Buffer settings
	BufferSize       int           `yaml:"buffer_size"`
	FlushInterval    time.Duration `yaml:"flush_interval"`
	MaxMessageSize   int64         `yaml:"max_message_size"`
	CompressionLevel int           `yaml:"compression_level"`
	
	// Connection management
	ClientTimeout    time.Duration `yaml:"client_timeout"`
	PingInterval     time.Duration `yaml:"ping_interval"`
	MaxClients       int           `yaml:"max_clients"`
}

type WebSocketConfig struct {
	ReadBufferSize    int           `yaml:"read_buffer_size"`
	WriteBufferSize   int           `yaml:"write_buffer_size"`
	HandshakeTimeout  time.Duration `yaml:"handshake_timeout"`
	EnableCompression bool          `yaml:"enable_compression"`
	CheckOrigin       bool          `yaml:"check_origin"`
	Subprotocols      []string      `yaml:"subprotocols"`
}

type SSEConfig struct {
	RetryInterval     time.Duration `yaml:"retry_interval"`
	MaxEventSize      int           `yaml:"max_event_size"`
	HeartbeatInterval time.Duration `yaml:"heartbeat_interval"`
	EnableCORS        bool          `yaml:"enable_cors"`
	AllowedOrigins    []string      `yaml:"allowed_origins"`
}

type MetricsConfig struct {
	Enabled bool   `yaml:"enabled"`
	Port    int    `yaml:"port"`
	Path    string `yaml:"path"`
}

type SecurityConfig struct {
	RateLimit RateLimitConfig `yaml:"rate_limit"`
	CORS      CORSConfig      `yaml:"cors"`
	Auth      AuthConfig      `yaml:"auth"`
}

type RateLimitConfig struct {
	Enabled           bool          `yaml:"enabled"`
	RequestsPerSecond int           `yaml:"requests_per_second"`
	BurstSize         int           `yaml:"burst_size"`
	CleanupInterval   time.Duration `yaml:"cleanup_interval"`
}

type CORSConfig struct {
	Enabled        bool     `yaml:"enabled"`
	AllowedOrigins []string `yaml:"allowed_origins"`
	AllowedMethods []string `yaml:"allowed_methods"`
	AllowedHeaders []string `yaml:"allowed_headers"`
	MaxAge         int      `yaml:"max_age"`
}

type AuthConfig struct {
	Enabled   bool   `yaml:"enabled"`
	HeaderKey string `yaml:"header_key"`
	TokenType string `yaml:"token_type"`
}

func Load() *Config {
	return &Config{
		Server: ServerConfig{
			Host:            getEnv("SERVER_HOST", "0.0.0.0"),
			Port:            getEnvInt("SERVER_PORT", 8080),
			ReadTimeout:     getDurationEnv("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout:    getDurationEnv("SERVER_WRITE_TIMEOUT", 30*time.Second),
			IdleTimeout:     getDurationEnv("SERVER_IDLE_TIMEOUT", 120*time.Second),
			MaxConnections:  getEnvInt("SERVER_MAX_CONNECTIONS", 1000),
			EnableProfiling: getEnvBool("SERVER_ENABLE_PROFILING", false),
		},
		Backend: BackendConfig{
			BaseURL:             getEnv("BACKEND_BASE_URL", "http://localhost:8000"),
			Timeout:             getDurationEnv("BACKEND_TIMEOUT", 30*time.Second),
			MaxIdleConns:        getEnvInt("BACKEND_MAX_IDLE_CONNS", 100),
			MaxConnsPerHost:     getEnvInt("BACKEND_MAX_CONNS_PER_HOST", 10),
			IdleConnTimeout:     getDurationEnv("BACKEND_IDLE_CONN_TIMEOUT", 90*time.Second),
			TLSHandshakeTimeout: getDurationEnv("BACKEND_TLS_HANDSHAKE_TIMEOUT", 10*time.Second),
			KeepAlive:           getDurationEnv("BACKEND_KEEP_ALIVE", 30*time.Second),
			MaxRetries:          getEnvInt("BACKEND_MAX_RETRIES", 3),
			RetryDelay:          getDurationEnv("BACKEND_RETRY_DELAY", 100*time.Millisecond),
		},
		Streaming: StreamingConfig{
			WebSocket: WebSocketConfig{
				ReadBufferSize:    getEnvInt("WS_READ_BUFFER_SIZE", 4096),
				WriteBufferSize:   getEnvInt("WS_WRITE_BUFFER_SIZE", 4096),
				HandshakeTimeout:  getDurationEnv("WS_HANDSHAKE_TIMEOUT", 10*time.Second),
				EnableCompression: getEnvBool("WS_ENABLE_COMPRESSION", true),
				CheckOrigin:       getEnvBool("WS_CHECK_ORIGIN", false),
				Subprotocols:      getEnvSlice("WS_SUBPROTOCOLS", []string{"smartrag-stream"}),
			},
			SSE: SSEConfig{
				RetryInterval:     getDurationEnv("SSE_RETRY_INTERVAL", 5*time.Second),
				MaxEventSize:      getEnvInt("SSE_MAX_EVENT_SIZE", 65536),
				HeartbeatInterval: getDurationEnv("SSE_HEARTBEAT_INTERVAL", 30*time.Second),
				EnableCORS:        getEnvBool("SSE_ENABLE_CORS", true),
				AllowedOrigins:    getEnvSlice("SSE_ALLOWED_ORIGINS", []string{"*"}),
			},
			BufferSize:       getEnvInt("STREAMING_BUFFER_SIZE", 8192),
			FlushInterval:    getDurationEnv("STREAMING_FLUSH_INTERVAL", 100*time.Millisecond),
			MaxMessageSize:   int64(getEnvInt("STREAMING_MAX_MESSAGE_SIZE", 1048576)), // 1MB
			CompressionLevel: getEnvInt("STREAMING_COMPRESSION_LEVEL", 6),
			ClientTimeout:    getDurationEnv("STREAMING_CLIENT_TIMEOUT", 300*time.Second),
			PingInterval:     getDurationEnv("STREAMING_PING_INTERVAL", 30*time.Second),
			MaxClients:       getEnvInt("STREAMING_MAX_CLIENTS", 1000),
		},
		Metrics: MetricsConfig{
			Enabled: getEnvBool("METRICS_ENABLED", true),
			Port:    getEnvInt("METRICS_PORT", 9090),
			Path:    getEnv("METRICS_PATH", "/metrics"),
		},
		Security: SecurityConfig{
			RateLimit: RateLimitConfig{
				Enabled:           getEnvBool("RATE_LIMIT_ENABLED", true),
				RequestsPerSecond: getEnvInt("RATE_LIMIT_RPS", 100),
				BurstSize:         getEnvInt("RATE_LIMIT_BURST", 200),
				CleanupInterval:   getDurationEnv("RATE_LIMIT_CLEANUP_INTERVAL", 60*time.Second),
			},
			CORS: CORSConfig{
				Enabled:        getEnvBool("CORS_ENABLED", true),
				AllowedOrigins: getEnvSlice("CORS_ALLOWED_ORIGINS", []string{"*"}),
				AllowedMethods: getEnvSlice("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
				AllowedHeaders: getEnvSlice("CORS_ALLOWED_HEADERS", []string{"*"}),
				MaxAge:         getEnvInt("CORS_MAX_AGE", 86400),
			},
			Auth: AuthConfig{
				Enabled:   getEnvBool("AUTH_ENABLED", false),
				HeaderKey: getEnv("AUTH_HEADER_KEY", "Authorization"),
				TokenType: getEnv("AUTH_TOKEN_TYPE", "Bearer"),
			},
		},
	}
}

// Helper functions
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getEnvSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		// Simple comma-separated parsing
		result := make([]string, 0)
		for _, item := range strings.Split(value, ",") {
			if trimmed := strings.TrimSpace(item); trimmed != "" {
				result = append(result, trimmed)
			}
		}
		if len(result) > 0 {
			return result
		}
	}
	return defaultValue
}