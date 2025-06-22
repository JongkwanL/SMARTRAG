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
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Load configuration
	cfg := config.Load()

	// Initialize Redis for rate limiting and caching
	redisClient := cache.NewRedisClient(cfg.Redis.Address, cfg.Redis.Password, cfg.Redis.DB)

	// Test Redis connection
	if err := redisClient.Ping(context.Background()); err != nil {
		logger.Fatal("Failed to connect to Redis", zap.Error(err))
	}

	// Initialize rate limiter
	rateLimiter := ratelimit.NewRedisRateLimiter(
		redisClient.GetClient(),
		cfg.RateLimit.RequestsPerMinute,
		time.Duration(cfg.RateLimit.WindowSize)*time.Second,
	)

	// Initialize auth service
	authService := auth.NewJWTAuth(cfg.Auth.SecretKey, cfg.Auth.TokenTTL)

	// Initialize HTTP proxy
	httpProxy, err := proxy.NewHandler(
		cfg.Backend.URL,
		time.Duration(cfg.Backend.ResponseTimeout)*time.Second,
		logger,
	)
	if err != nil {
		logger.Fatal("Failed to create proxy handler", zap.Error(err))
	}

	// Setup Gin router
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(middleware.CORS(cfg.CORS))
	router.Use(middleware.Logging(logger))
	router.Use(middleware.Metrics())

	// Health check endpoint
	router.GET("/health", httpProxy.HealthCheck)

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
				httpProxy.ProxyRequest,
			)

			// Query endpoint (with aggressive caching)
			protected.POST("/query",
				middleware.Cache(redisClient, 60), // 1 min cache
				httpProxy.ProxyRequest,
			)

			// All other endpoints proxy to Python backend
			protected.Any("/*path", httpProxy.ProxyRequest)
		}
	}

	// WebSocket endpoints for real-time streaming (TODO: implement WebSocket manager)
	// router.GET("/ws/stream", wsManager.HandleConnection)

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