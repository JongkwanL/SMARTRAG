package main

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"smartrag-monitoring-agent/pkg/types"
)

// setupRouter configures the HTTP router with all endpoints
func setupRouter(agent *MonitoringAgent) *gin.Engine {
	router := gin.New()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(loggingMiddleware(agent.logger))
	
	if agent.config.Server.EnableCORS {
		router.Use(corsMiddleware())
	}

	// Health and status endpoints
	router.GET("/health", healthHandler(agent))
	router.GET("/ready", readinessHandler(agent))
	router.GET("/status", statusHandler(agent))

	// Metrics endpoints
	router.GET("/metrics/system", systemMetricsHandler(agent))
	router.GET("/metrics/health", healthMetricsHandler(agent))
	router.GET("/metrics/stats", statsHandler(agent))

	// API v1 endpoints
	v1 := router.Group("/api/v1")
	{
		v1.GET("/metrics", getAllMetricsHandler(agent))
		v1.GET("/metrics/system", systemMetricsHandler(agent))
		v1.GET("/metrics/health", healthMetricsHandler(agent))
		v1.GET("/health", healthCheckHandler(agent))
		v1.GET("/stats", statsHandler(agent))
		v1.GET("/config", configHandler(agent))
		v1.POST("/health/check", triggerHealthCheckHandler(agent))
	}

	// Admin endpoints (if pprof is enabled)
	if agent.config.Server.EnablePprof {
		router.GET("/debug/pprof/*path", gin.WrapH(http.DefaultServeMux))
	}

	return router
}

// healthHandler returns basic health status
func healthHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		status := "healthy"
		uptime := time.Since(agent.startTime)

		c.JSON(http.StatusOK, gin.H{
			"status":    status,
			"timestamp": time.Now().Format(time.RFC3339),
			"uptime":    uptime.String(),
			"version":   version,
		})
	}
}

// readinessHandler returns readiness status
func readinessHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		ready := agent.config.Monitoring.Enabled
		status := http.StatusOK

		if !ready {
			status = http.StatusServiceUnavailable
		}

		c.JSON(status, gin.H{
			"ready":     ready,
			"timestamp": time.Now().Format(time.RFC3339),
			"message":   "Monitoring agent ready",
		})
	}
}

// statusHandler returns detailed status information
func statusHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		uptime := time.Since(agent.startTime)
		
		// Get health collector stats
		var healthStats *collectors.HealthStats
		if agent.healthCollector != nil {
			healthStats = agent.healthCollector.GetStats()
		}

		status := gin.H{
			"status":     "healthy",
			"timestamp":  time.Now().Format(time.RFC3339),
			"uptime":     uptime.String(),
			"version":    version,
			"commit":     commit,
			"build_date": date,
			"monitoring": gin.H{
				"enabled":                agent.config.Monitoring.Enabled,
				"collection_interval":    agent.config.Monitoring.CollectionInterval.String(),
				"health_check_interval":  agent.config.Monitoring.HealthCheckInterval.String(),
				"process_monitoring":     agent.config.Monitoring.ProcessMonitoring,
				"temperature_monitoring": agent.config.Monitoring.TemperatureMonitoring,
				"service_count":          len(agent.config.Monitoring.Services),
			},
			"stats": agent.stats,
		}

		if healthStats != nil {
			status["health_stats"] = healthStats
		}

		c.JSON(http.StatusOK, status)
	}
}

// systemMetricsHandler returns current system metrics
func systemMetricsHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !agent.config.Monitoring.Enabled || agent.systemCollector == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": "System monitoring is disabled",
			})
			return
		}

		// Collect current metrics
		ctx := c.Request.Context()
		metrics, err := agent.systemCollector.Collect(ctx)
		if err != nil {
			agent.logger.Error("Failed to collect system metrics for API", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to collect system metrics",
			})
			return
		}

		c.JSON(http.StatusOK, metrics)
	}
}

// healthMetricsHandler returns current health check results
func healthMetricsHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		if agent.healthCollector == nil || len(agent.config.Monitoring.Services) == 0 {
			c.JSON(http.StatusOK, []types.ServiceHealth{})
			return
		}

		// Perform health checks
		ctx := c.Request.Context()
		healthResults, err := agent.healthCollector.CheckAll(ctx)
		if err != nil {
			agent.logger.Error("Failed to perform health checks for API", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to perform health checks",
			})
			return
		}

		c.JSON(http.StatusOK, healthResults)
	}
}

// getAllMetricsHandler returns all metrics (system + health)
func getAllMetricsHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		result := gin.H{
			"timestamp": time.Now().Format(time.RFC3339),
		}

		// Get system metrics if enabled
		if agent.config.Monitoring.Enabled && agent.systemCollector != nil {
			ctx := c.Request.Context()
			systemMetrics, err := agent.systemCollector.Collect(ctx)
			if err != nil {
				agent.logger.Warn("Failed to collect system metrics for all metrics API", zap.Error(err))
			} else {
				result["system"] = systemMetrics
			}
		}

		// Get health metrics if enabled
		if agent.healthCollector != nil && len(agent.config.Monitoring.Services) > 0 {
			ctx := c.Request.Context()
			healthResults, err := agent.healthCollector.CheckAll(ctx)
			if err != nil {
				agent.logger.Warn("Failed to perform health checks for all metrics API", zap.Error(err))
			} else {
				result["health"] = healthResults
			}
		}

		c.JSON(http.StatusOK, result)
	}
}

// statsHandler returns monitoring statistics
func statsHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		stats := *agent.stats
		stats.UptimeSeconds = int64(time.Since(agent.startTime).Seconds())

		// Add health collector stats if available
		if agent.healthCollector != nil {
			healthStats := agent.healthCollector.GetStats()
			stats.HealthChecksRun = healthStats.TotalChecks
			stats.HealthCheckErrors = healthStats.FailedChecks
		}

		c.JSON(http.StatusOK, stats)
	}
}

// configHandler returns current configuration (sanitized)
func configHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Return sanitized configuration (without sensitive data)
		config := gin.H{
			"server": gin.H{
				"host":         agent.config.Server.Host,
				"port":         agent.config.Server.Port,
				"metrics_port": agent.config.Server.MetricsPort,
				"enable_pprof": agent.config.Server.EnablePprof,
				"enable_cors":  agent.config.Server.EnableCORS,
			},
			"monitoring": gin.H{
				"enabled":                  agent.config.Monitoring.Enabled,
				"collection_interval":      agent.config.Monitoring.CollectionInterval.String(),
				"health_check_interval":    agent.config.Monitoring.HealthCheckInterval.String(),
				"process_monitoring":       agent.config.Monitoring.ProcessMonitoring,
				"temperature_monitoring":   agent.config.Monitoring.TemperatureMonitoring,
				"network_interfaces":       agent.config.Monitoring.NetworkInterfaces,
				"disk_mountpoints":         agent.config.Monitoring.DiskMountpoints,
				"process_filters":          agent.config.Monitoring.ProcessFilters,
				"service_count":            len(agent.config.Monitoring.Services),
			},
			"storage": gin.H{
				"type":              agent.config.Storage.Type,
				"metrics_retention": agent.config.Storage.MetricsRetention.String(),
				"alerts_retention":  agent.config.Storage.AlertsRetention.String(),
				"health_retention":  agent.config.Storage.HealthRetention.String(),
			},
			"alerts": gin.H{
				"enabled":    agent.config.Alerts.Enabled,
				"rule_count": len(agent.config.Alerts.Rules),
			},
		}

		c.JSON(http.StatusOK, config)
	}
}

// healthCheckHandler performs an immediate health check
func healthCheckHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		if agent.healthCollector == nil || len(agent.config.Monitoring.Services) == 0 {
			c.JSON(http.StatusOK, gin.H{
				"message": "No services configured for health checks",
				"results": []types.ServiceHealth{},
			})
			return
		}

		ctx := c.Request.Context()
		healthResults, err := agent.healthCollector.CheckAll(ctx)
		if err != nil {
			agent.logger.Error("Failed to perform immediate health check", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to perform health checks",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message":   "Health check completed",
			"timestamp": time.Now().Format(time.RFC3339),
			"results":   healthResults,
		})
	}
}

// triggerHealthCheckHandler triggers an immediate health check (POST)
func triggerHealthCheckHandler(agent *MonitoringAgent) gin.HandlerFunc {
	return func(c *gin.Context) {
		if agent.healthCollector == nil || len(agent.config.Monitoring.Services) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "No services configured for health checks",
			})
			return
		}

		// Parse request body for specific services (optional)
		var request struct {
			Services []string `json:"services,omitempty"`
		}
		
		if err := c.ShouldBindJSON(&request); err != nil {
			// Ignore JSON parsing errors, just check all services
		}

		ctx := c.Request.Context()
		healthResults, err := agent.healthCollector.CheckAll(ctx)
		if err != nil {
			agent.logger.Error("Failed to trigger health check", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to perform health checks",
			})
			return
		}

		// Filter results if specific services were requested
		if len(request.Services) > 0 {
			filteredResults := make([]types.ServiceHealth, 0, len(request.Services))
			for _, result := range healthResults {
				for _, serviceName := range request.Services {
					if result.ServiceName == serviceName {
						filteredResults = append(filteredResults, result)
						break
					}
				}
			}
			healthResults = filteredResults
		}

		c.JSON(http.StatusOK, gin.H{
			"message":   "Health check triggered successfully",
			"timestamp": time.Now().Format(time.RFC3339),
			"results":   healthResults,
		})
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

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Requested-With")
		c.Header("Access-Control-Max-Age", "3600")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}