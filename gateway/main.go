package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"smartrag-gateway/internal/auth"
	"smartrag-gateway/internal/cache"
	"smartrag-gateway/internal/config"
	"smartrag-gateway/internal/middleware"
	"smartrag-gateway/internal/proxy"
	"smartrag-gateway/internal/ratelimit"
	"smartrag-gateway/internal/websocket"
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Load configuration
	cfg := config.Load()

	// Initialize Redis for rate limiting and caching
	redisClient := cache.NewRedisClient(cfg.Redis)

	// Initialize rate limiter
	rateLimiter := ratelimit.NewRedisRateLimiter(redisClient, cfg.RateLimit)

	// Initialize auth service
	authService := auth.NewJWTAuth(cfg.Auth.SecretKey)

	// Initialize HTTP proxy
	httpProxy := proxy.NewHTTPProxy(cfg.Backend)

	// Initialize WebSocket manager
	wsManager := websocket.NewManager(logger)

	// Setup Gin router
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(middleware.CORS(cfg.CORS))
	router.Use(middleware.Logging(logger))
	router.Use(middleware.Metrics())

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
			"service":   "smartrag-gateway",
		})
	})

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(middleware.PrometheusHandler()))

	// API routes with middleware
	api := router.Group("/api/v1")
	{
		// Apply rate limiting
		api.Use(middleware.RateLimit(rateLimiter))

		// Public endpoints (no auth required)
		public := api.Group("")
		{
			public.GET("/health", func(c *gin.Context) {
				c.JSON(200, gin.H{"status": "ok"})
			})
		}

		// Protected endpoints (auth required)
		protected := api.Group("")
		protected.Use(middleware.Auth(authService))
		{
			// Document ingestion (with caching)
			protected.POST("/ingest", 
				middleware.Cache(redisClient, 300), // 5 min cache
				httpProxy.ProxyHTTP,
			)

			// Query endpoint (with aggressive caching)
			protected.POST("/query",
				middleware.Cache(redisClient, 60), // 1 min cache
				httpProxy.ProxyHTTP,
			)

			// All other endpoints proxy to Python backend
			protected.Any("/*path", httpProxy.ProxyHTTP)
		}
	}

	// WebSocket endpoints for real-time streaming
	router.GET("/ws/stream", wsManager.HandleConnection)

	// Start server
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(cfg.Server.IdleTimeout) * time.Second,
	}

	// Graceful shutdown
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	logger.Info("SmartRAG Gateway started", 
		zap.Int("port", cfg.Server.Port),
		zap.String("env", cfg.Environment),
	)

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Server exited")
}