package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	"golang.org/x/time/rate"

	"smartrag-streaming-proxy/internal/config"
	"smartrag-streaming-proxy/internal/metrics"
	"smartrag-streaming-proxy/internal/proxy"
	"smartrag-streaming-proxy/internal/sse"
	"smartrag-streaming-proxy/internal/websocket"
	"smartrag-streaming-proxy/pkg/types"
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
		zap.String("backend_url", cfg.Backend.BaseURL),
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

	// Initialize backend proxy
	streamingProxy := proxy.NewStreamingProxy(&cfg.Backend, logger)
	logger.Info("Streaming proxy initialized")

	// Test backend connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := streamingProxy.HealthCheck(ctx); err != nil {
		logger.Warn("Backend health check failed", zap.Error(err))
	} else {
		logger.Info("Backend connectivity verified")
	}
	cancel()

	// Initialize WebSocket manager
	wsManager := websocket.NewManager(&cfg.Streaming, logger, streamingProxy)
	wsManager.Start()
	defer wsManager.Stop()

	// Initialize SSE handler
	sseHandler := sse.NewHandler(&cfg.Streaming.SSE, logger, streamingProxy)

	// Start system metrics collection
	go collectSystemMetrics(metricsCollector, logger)

	// Set Gin mode based on environment
	if cfg.Server.EnableProfiling {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create Gin router
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(loggingMiddleware(logger))
	router.Use(corsMiddleware(&cfg.Security.CORS))

	// Add rate limiting if enabled
	if cfg.Security.RateLimit.Enabled {
		limiter := rate.NewLimiter(
			rate.Limit(cfg.Security.RateLimit.RequestsPerSecond),
			cfg.Security.RateLimit.BurstSize,
		)
		router.Use(rateLimitMiddleware(limiter))
	}

	// Health and status endpoints
	router.GET("/health", healthHandler(streamingProxy, logger))
	router.GET("/status", statusHandler(wsManager, sseHandler, streamingProxy))

	// WebSocket endpoints
	router.GET("/ws", wsManager.HandleConnection)
	router.GET("/websocket", wsManager.HandleConnection)

	// Server-Sent Events endpoints
	router.GET("/sse", sseHandler.HandleConnection)
	router.POST("/sse/stream", sseHandler.HandleStreamingQuery)

	// API endpoints
	api := router.Group("/api/v1")
	{
		api.GET("/stats", statsHandler(wsManager, sseHandler, streamingProxy))
		api.POST("/broadcast", broadcastHandler(wsManager, sseHandler))
	}

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
		MaxHeaderBytes: 1 << 20, // 1MB
	}

	// Start server in goroutine
	go func() {
		logger.Info("Starting streaming proxy server",
			zap.String("addr", server.Addr),
			zap.Bool("metrics_enabled", cfg.Metrics.Enabled),
		)
		
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Server failed to start", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigChan

	logger.Info("Received shutdown signal", zap.String("signal", sig.String()))

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server shutdown error", zap.Error(err))
	} else {
		logger.Info("Server shut down gracefully")
	}
}

// Middleware functions

func loggingMiddleware(logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		c.Next()

		latency := time.Since(start)
		
		logger.Info("Request completed",
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.String("query", query),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("latency", latency),
			zap.String("client_ip", c.ClientIP()),
			zap.String("user_agent", c.Request.UserAgent()),
		)
	}
}

func corsMiddleware(cfg *config.CORSConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !cfg.Enabled {
			c.Next()
			return
		}

		origin := c.Request.Header.Get("Origin")
		
		// Check allowed origins
		allowed := false
		for _, allowedOrigin := range cfg.AllowedOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.Header("Access-Control-Allow-Methods", joinStrings(cfg.AllowedMethods, ", "))
		c.Header("Access-Control-Allow-Headers", joinStrings(cfg.AllowedHeaders, ", "))
		c.Header("Access-Control-Max-Age", fmt.Sprintf("%d", cfg.MaxAge))
		c.Header("Access-Control-Allow-Credentials", "true")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func rateLimitMiddleware(limiter *rate.Limiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !limiter.Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded",
				"retry_after": limiter.Reserve().Delay().Seconds(),
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

// Handler functions

func healthHandler(proxy *proxy.StreamingProxy, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Check backend health
		backendStatus := "healthy"
		if err := proxy.HealthCheck(ctx); err != nil {
			backendStatus = "unhealthy"
			logger.Warn("Backend health check failed", zap.Error(err))
		}

		status := "healthy"
		if backendStatus != "healthy" {
			status = "degraded"
		}

		c.JSON(http.StatusOK, gin.H{
			"status": status,
			"timestamp": time.Now().Format(time.RFC3339),
			"version": "1.0.0",
			"backend": backendStatus,
		})
	}
}

func statusHandler(wsManager *websocket.Manager, sseHandler *sse.Handler, proxy *proxy.StreamingProxy) gin.HandlerFunc {
	return func(c *gin.Context) {
		wsStats := wsManager.GetStats()
		sseStats := sseHandler.GetStats()
		proxyStats := proxy.GetStats()

		c.JSON(http.StatusOK, gin.H{
			"websocket": wsStats,
			"sse": sseStats,
			"proxy": proxyStats,
			"timestamp": time.Now().Format(time.RFC3339),
		})
	}
}

func statsHandler(wsManager *websocket.Manager, sseHandler *sse.Handler, proxy *proxy.StreamingProxy) gin.HandlerFunc {
	return func(c *gin.Context) {
		wsStats := wsManager.GetStats()
		sseStats := sseHandler.GetStats()
		proxyStats := proxy.GetStats()

		totalConnections := wsStats.ActiveConnections + sseStats.ActiveConnections
		totalMessages := wsStats.MessagesProcessed + sseStats.EventsSent
		totalBytes := wsStats.BytesTransferred + sseStats.BytesTransferred

		c.JSON(http.StatusOK, types.ProxyStats{
			ActiveConnections: totalConnections,
			TotalConnections:  wsStats.TotalConnections + sseStats.TotalConnections,
			MessagesProcessed: totalMessages,
			BytesTransferred:  totalBytes,
			ErrorCount:        wsStats.ErrorCount + sseStats.ErrorCount,
			ConnectionsByType: map[string]int{
				"websocket": wsStats.ActiveConnections,
				"sse":       sseStats.ActiveConnections,
			},
		})
	}
}

func broadcastHandler(wsManager *websocket.Manager, sseHandler *sse.Handler) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request struct {
			Event string      `json:"event"`
			Data  interface{} `json:"data"`
		}

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Invalid request format",
			})
			return
		}

		// Broadcast to WebSocket clients
		// This would need to be implemented in the WebSocket manager

		// Broadcast to SSE clients
		sseEvent := &sse.Event{
			Event: request.Event,
			Data:  fmt.Sprintf("%v", request.Data),
		}
		sseHandler.BroadcastEvent(sseEvent)

		c.JSON(http.StatusOK, gin.H{
			"message": "Broadcast sent",
		})
	}
}

// collectSystemMetrics collects system metrics periodically
func collectSystemMetrics(metrics *metrics.Metrics, logger *zap.Logger) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Get memory stats
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)

		// Update metrics
		metrics.SetMemoryUsage(memStats.Alloc)
		metrics.SetGoroutines(runtime.NumGoroutine())

		logger.Debug("System metrics collected",
			zap.Uint64("memory_alloc", memStats.Alloc),
			zap.Int("goroutines", runtime.NumGoroutine()),
		)
	}
}

// Helper functions

func joinStrings(slice []string, separator string) string {
	if len(slice) == 0 {
		return ""
	}
	
	result := slice[0]
	for i := 1; i < len(slice); i++ {
		result += separator + slice[i]
	}
	return result
}