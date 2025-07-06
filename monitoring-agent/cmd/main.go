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

	"smartrag-monitoring-agent/internal/collectors"
	"smartrag-monitoring-agent/internal/config"
	"smartrag-monitoring-agent/internal/metrics"
	"smartrag-monitoring-agent/pkg/types"
)

var (
	version = "1.0.0"
	commit  = "unknown"
	date    = "unknown"
)

func main() {
	// Initialize logger
	logger, err := initLogger()
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize logger: %v", err))
	}
	defer logger.Sync()

	logger.Info("Starting SmartRAG Monitoring Agent",
		zap.String("version", version),
		zap.String("commit", commit),
		zap.String("build_date", date),
		zap.String("go_version", runtime.Version()),
	)

	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	logger.Info("Configuration loaded",
		zap.String("server_host", cfg.Server.Host),
		zap.Int("server_port", cfg.Server.Port),
		zap.Int("metrics_port", cfg.Server.MetricsPort),
		zap.Bool("monitoring_enabled", cfg.Monitoring.Enabled),
		zap.Duration("collection_interval", cfg.Monitoring.CollectionInterval),
		zap.Int("service_count", len(cfg.Monitoring.Services)),
	)

	// Initialize Prometheus metrics
	prometheusMetrics := metrics.NewPrometheusMetrics()
	logger.Info("Prometheus metrics initialized")

	// Start metrics server if enabled
	go func() {
		addr := fmt.Sprintf(":%d", cfg.Server.MetricsPort)
		logger.Info("Starting metrics server", zap.String("addr", addr))
		if err := prometheusMetrics.StartMetricsServer(addr); err != nil {
			logger.Error("Metrics server failed", zap.Error(err))
		}
	}()

	// Initialize collectors
	var systemCollector *collectors.SystemCollector
	var healthCollector *collectors.HealthCollector

	if cfg.Monitoring.Enabled {
		// Initialize system collector
		systemCollector = collectors.NewSystemCollector(
			logger,
			cfg.Monitoring.ProcessMonitoring,
			cfg.Monitoring.ProcessFilters,
			cfg.Monitoring.NetworkInterfaces,
			cfg.Monitoring.DiskMountpoints,
			cfg.Monitoring.TemperatureMonitoring,
		)

		// Initialize health collector
		healthCollector = collectors.NewHealthCollector(
			logger,
			cfg.Monitoring.Services,
		)

		logger.Info("Collectors initialized",
			zap.Bool("system_enabled", true),
			zap.Bool("health_enabled", len(cfg.Monitoring.Services) > 0),
			zap.Bool("process_monitoring", cfg.Monitoring.ProcessMonitoring),
			zap.Bool("temperature_monitoring", cfg.Monitoring.TemperatureMonitoring),
		)
	}

	// Initialize monitoring agent
	agent := &MonitoringAgent{
		config:            cfg,
		logger:            logger,
		systemCollector:   systemCollector,
		healthCollector:   healthCollector,
		prometheusMetrics: prometheusMetrics,
		startTime:         time.Now(),
		stats:             &types.MonitoringStats{
			StartTime: time.Now(),
		},
	}

	// Start monitoring loops
	if cfg.Monitoring.Enabled {
		agent.startMonitoring()
	}

	// Set Gin mode
	if logger.Core().Enabled(zap.DebugLevel) {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create HTTP server
	router := setupRouter(agent)
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
		logger.Info("Starting HTTP server",
			zap.String("addr", server.Addr),
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

	// Stop monitoring
	agent.stopMonitoring()
}

// MonitoringAgent represents the main monitoring agent
type MonitoringAgent struct {
	config            *config.Config
	logger            *zap.Logger
	systemCollector   *collectors.SystemCollector
	healthCollector   *collectors.HealthCollector
	prometheusMetrics *metrics.PrometheusMetrics
	startTime         time.Time
	stats             *types.MonitoringStats

	// Control channels
	stopChan   chan struct{}
	systemDone chan struct{}
	healthDone chan struct{}
}

// startMonitoring starts the monitoring loops
func (a *MonitoringAgent) startMonitoring() {
	a.stopChan = make(chan struct{})
	a.systemDone = make(chan struct{})
	a.healthDone = make(chan struct{})

	// Start system metrics collection
	if a.systemCollector != nil {
		go a.systemMetricsLoop()
	}

	// Start health checks
	if a.healthCollector != nil && len(a.config.Monitoring.Services) > 0 {
		go a.healthCheckLoop()
	}

	a.logger.Info("Monitoring started")
}

// stopMonitoring stops the monitoring loops
func (a *MonitoringAgent) stopMonitoring() {
	if a.stopChan != nil {
		close(a.stopChan)

		// Wait for loops to finish
		if a.systemCollector != nil {
			<-a.systemDone
		}
		if a.healthCollector != nil && len(a.config.Monitoring.Services) > 0 {
			<-a.healthDone
		}
	}

	a.logger.Info("Monitoring stopped")
}

// systemMetricsLoop runs the system metrics collection loop
func (a *MonitoringAgent) systemMetricsLoop() {
	defer close(a.systemDone)

	ticker := time.NewTicker(a.config.Monitoring.CollectionInterval)
	defer ticker.Stop()

	a.logger.Info("Starting system metrics collection",
		zap.Duration("interval", a.config.Monitoring.CollectionInterval),
	)

	for {
		select {
		case <-ticker.C:
			a.collectSystemMetrics()
		case <-a.stopChan:
			return
		}
	}
}

// healthCheckLoop runs the health check loop
func (a *MonitoringAgent) healthCheckLoop() {
	defer close(a.healthDone)

	ticker := time.NewTicker(a.config.Monitoring.HealthCheckInterval)
	defer ticker.Stop()

	a.logger.Info("Starting health checks",
		zap.Duration("interval", a.config.Monitoring.HealthCheckInterval),
		zap.Int("service_count", len(a.config.Monitoring.Services)),
	)

	for {
		select {
		case <-ticker.C:
			a.performHealthChecks()
		case <-a.stopChan:
			return
		}
	}
}

// collectSystemMetrics collects and updates system metrics
func (a *MonitoringAgent) collectSystemMetrics() {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		a.prometheusMetrics.ObserveCollectionDuration(duration.Seconds())
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	metrics, err := a.systemCollector.Collect(ctx)
	if err != nil {
		a.logger.Error("Failed to collect system metrics", zap.Error(err))
		a.prometheusMetrics.IncCollectionErrors("system")
		return
	}

	// Update Prometheus metrics
	a.updatePrometheusSystemMetrics(metrics)
	a.prometheusMetrics.IncMetricsCollected()

	// Update stats
	a.stats.MetricsCollected++
	a.stats.LastCollectionTime = time.Now()

	a.logger.Debug("System metrics collected",
		zap.Duration("duration", time.Since(start)),
		zap.Float64("cpu_percent", metrics.CPU.UsagePercent),
		zap.Float64("memory_percent", metrics.Memory.UsagePercent),
		zap.Int("disk_count", len(metrics.Disk)),
		zap.Int("network_count", len(metrics.Network)),
	)
}

// performHealthChecks performs health checks on all services
func (a *MonitoringAgent) performHealthChecks() {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		a.prometheusMetrics.ObserveHealthCheckDuration(duration.Seconds())
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	healthResults, err := a.healthCollector.CheckAll(ctx)
	if err != nil {
		a.logger.Error("Failed to perform health checks", zap.Error(err))
		a.prometheusMetrics.IncCollectionErrors("health")
		return
	}

	// Update Prometheus metrics
	for _, health := range healthResults {
		a.prometheusMetrics.UpdateServiceHealth(
			health.ServiceName,
			health.Endpoint,
			health.CheckType,
			string(health.Status),
			health.ResponseTime.Seconds(),
		)
	}

	// Update stats
	a.stats.HealthChecksRun++

	a.logger.Debug("Health checks completed",
		zap.Duration("duration", time.Since(start)),
		zap.Int("service_count", len(healthResults)),
	)
}

// updatePrometheusSystemMetrics updates Prometheus metrics with system data
func (a *MonitoringAgent) updatePrometheusSystemMetrics(m *types.SystemMetrics) {
	// Update basic system metrics
	a.prometheusMetrics.UpdateSystemMetrics(
		m.CPU.UsagePercent,
		m.Memory.UsagePercent,
		float64(m.Memory.UsedBytes),
		float64(m.Memory.TotalBytes),
	)

	// Update load metrics
	a.prometheusMetrics.UpdateLoadMetrics(
		m.Load.Load1,
		m.Load.Load5,
		m.Load.Load15,
	)

	// Update disk metrics
	for _, disk := range m.Disk {
		a.prometheusMetrics.UpdateDiskMetrics(
			disk.Device,
			disk.Mountpoint,
			disk.Filesystem,
			disk.UsagePercent,
			float64(disk.UsedBytes),
			float64(disk.TotalBytes),
		)
	}

	// Update network metrics
	for _, net := range m.Network {
		a.prometheusMetrics.UpdateNetworkMetrics(
			net.Interface,
			float64(net.BytesReceived),
			float64(net.BytesSent),
			float64(net.PacketsRecv),
			float64(net.PacketsSent),
			float64(net.ErrorsIn),
			float64(net.ErrorsOut),
		)
	}

	// Update process metrics
	if len(m.Process) > 0 {
		a.prometheusMetrics.UpdateProcessCount(float64(len(m.Process)))
		
		for _, proc := range m.Process {
			a.prometheusMetrics.UpdateProcessMetrics(
				proc.Name,
				fmt.Sprintf("%d", proc.PID),
				proc.Username,
				proc.CPUPercent,
				proc.MemoryPercent,
				float64(proc.MemoryRSS),
				float64(proc.MemoryVMS),
			)
		}
	}

	// Update temperature metrics
	for _, temp := range m.Temperature {
		a.prometheusMetrics.UpdateTemperatureMetrics(
			temp.SensorKey,
			temp.Temperature,
		)
	}

	// Update agent uptime
	uptime := time.Since(a.startTime).Seconds()
	a.prometheusMetrics.UpdateAgentMetrics(uptime)
	a.stats.UptimeSeconds = int64(uptime)
}

// initLogger initializes the logger
func initLogger() (*zap.Logger, error) {
	config := zap.NewProductionConfig()
	config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	
	// Set log level from environment
	if level := os.Getenv("LOG_LEVEL"); level != "" {
		var zapLevel zap.AtomicLevel
		if err := zapLevel.UnmarshalText([]byte(level)); err == nil {
			config.Level = zapLevel
		}
	}

	return config.Build()
}