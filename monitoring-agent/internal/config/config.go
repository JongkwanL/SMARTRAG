package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
	"smartrag-monitoring-agent/pkg/types"
)

// Config holds the complete configuration for the monitoring agent
type Config struct {
	Server     ServerConfig     `mapstructure:"server"`
	Monitoring MonitoringConfig `mapstructure:"monitoring"`
	Logging    LoggingConfig    `mapstructure:"logging"`
	Storage    StorageConfig    `mapstructure:"storage"`
	Alerts     AlertingConfig   `mapstructure:"alerts"`
	Security   SecurityConfig   `mapstructure:"security"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	MetricsPort  int           `mapstructure:"metrics_port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
	EnablePprof  bool          `mapstructure:"enable_pprof"`
	EnableCORS   bool          `mapstructure:"enable_cors"`
}

// MonitoringConfig holds monitoring collection settings
type MonitoringConfig struct {
	Enabled                bool                `mapstructure:"enabled"`
	CollectionInterval     time.Duration       `mapstructure:"collection_interval"`
	HealthCheckInterval    time.Duration       `mapstructure:"health_check_interval"`
	ProcessMonitoring      bool                `mapstructure:"process_monitoring"`
	ProcessFilters         []string            `mapstructure:"process_filters"`
	NetworkInterfaces      []string            `mapstructure:"network_interfaces"`
	DiskMountpoints        []string            `mapstructure:"disk_mountpoints"`
	TemperatureMonitoring  bool                `mapstructure:"temperature_monitoring"`
	Services               []ServiceConfig     `mapstructure:"services"`
	CustomMetrics          []CustomMetricConfig `mapstructure:"custom_metrics"`
}

// ServiceConfig defines a service to monitor
type ServiceConfig struct {
	Name           string            `mapstructure:"name"`
	Type           string            `mapstructure:"type"`
	Endpoint       string            `mapstructure:"endpoint"`
	Timeout        time.Duration     `mapstructure:"timeout"`
	Interval       time.Duration     `mapstructure:"interval"`
	Headers        map[string]string `mapstructure:"headers"`
	ExpectedStatus int               `mapstructure:"expected_status"`
	SSL            bool              `mapstructure:"ssl"`
	Enabled        bool              `mapstructure:"enabled"`
	Method         string            `mapstructure:"method"`
	Body           string            `mapstructure:"body"`
	FollowRedirects bool             `mapstructure:"follow_redirects"`
}

// CustomMetricConfig defines custom metrics to collect
type CustomMetricConfig struct {
	Name        string            `mapstructure:"name"`
	Type        string            `mapstructure:"type"` // command, file, http
	Source      string            `mapstructure:"source"`
	Interval    time.Duration     `mapstructure:"interval"`
	Parser      string            `mapstructure:"parser"` // json, regex, prometheus
	Pattern     string            `mapstructure:"pattern"`
	Labels      map[string]string `mapstructure:"labels"`
	Enabled     bool              `mapstructure:"enabled"`
}

// LoggingConfig holds logging configuration
type LoggingConfig struct {
	Level      string `mapstructure:"level"`
	Format     string `mapstructure:"format"` // json, console
	Output     string `mapstructure:"output"` // stdout, stderr, file
	Filename   string `mapstructure:"filename"`
	MaxSize    int    `mapstructure:"max_size_mb"`
	MaxBackups int    `mapstructure:"max_backups"`
	MaxAge     int    `mapstructure:"max_age_days"`
	Compress   bool   `mapstructure:"compress"`
}

// StorageConfig holds storage configuration
type StorageConfig struct {
	Type            string        `mapstructure:"type"` // memory, file, postgres
	Path            string        `mapstructure:"path"`
	MaxSizeMB       int64         `mapstructure:"max_size_mb"`
	DSN             string        `mapstructure:"dsn"`
	MetricsRetention time.Duration `mapstructure:"metrics_retention"`
	AlertsRetention  time.Duration `mapstructure:"alerts_retention"`
	HealthRetention  time.Duration `mapstructure:"health_retention"`
	CompressionEnabled bool        `mapstructure:"compression_enabled"`
	BackupEnabled    bool          `mapstructure:"backup_enabled"`
	BackupInterval   time.Duration `mapstructure:"backup_interval"`
}

// AlertingConfig holds alerting configuration
type AlertingConfig struct {
	Enabled     bool        `mapstructure:"enabled"`
	Rules       []AlertRule `mapstructure:"rules"`
	Webhooks    []Webhook   `mapstructure:"webhooks"`
	SMTPConfig  SMTPConfig  `mapstructure:"smtp"`
	SlackConfig SlackConfig `mapstructure:"slack"`
}

// AlertRule defines an alert rule
type AlertRule struct {
	Name        string            `mapstructure:"name"`
	Metric      string            `mapstructure:"metric"`
	Condition   string            `mapstructure:"condition"`
	Threshold   float64           `mapstructure:"threshold"`
	Duration    time.Duration     `mapstructure:"duration"`
	Severity    string            `mapstructure:"severity"`
	Labels      map[string]string `mapstructure:"labels"`
	Enabled     bool              `mapstructure:"enabled"`
	Description string            `mapstructure:"description"`
	Runbook     string            `mapstructure:"runbook"`
}

// Webhook defines a webhook endpoint for alerts
type Webhook struct {
	Name    string            `mapstructure:"name"`
	URL     string            `mapstructure:"url"`
	Method  string            `mapstructure:"method"`
	Headers map[string]string `mapstructure:"headers"`
	Timeout time.Duration     `mapstructure:"timeout"`
	Enabled bool              `mapstructure:"enabled"`
}

// SMTPConfig holds email notification settings
type SMTPConfig struct {
	Host     string   `mapstructure:"host"`
	Port     int      `mapstructure:"port"`
	Username string   `mapstructure:"username"`
	Password string   `mapstructure:"password"`
	From     string   `mapstructure:"from"`
	To       []string `mapstructure:"to"`
	UseTLS   bool     `mapstructure:"use_tls"`
	UseSSL   bool     `mapstructure:"use_ssl"`
}

// SlackConfig holds Slack notification settings
type SlackConfig struct {
	WebhookURL string `mapstructure:"webhook_url"`
	Channel    string `mapstructure:"channel"`
	Username   string `mapstructure:"username"`
	IconEmoji  string `mapstructure:"icon_emoji"`
}

// SecurityConfig holds security settings
type SecurityConfig struct {
	EnableAuth     bool              `mapstructure:"enable_auth"`
	APIKeys        []string          `mapstructure:"api_keys"`
	BasicAuth      BasicAuthConfig   `mapstructure:"basic_auth"`
	RateLimit      RateLimitConfig   `mapstructure:"rate_limit"`
	CORS           CORSConfig        `mapstructure:"cors"`
	TLS            TLSConfig         `mapstructure:"tls"`
	AllowedIPs     []string          `mapstructure:"allowed_ips"`
	BlockedIPs     []string          `mapstructure:"blocked_ips"`
}

// BasicAuthConfig holds basic authentication settings
type BasicAuthConfig struct {
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
}

// RateLimitConfig holds rate limiting settings
type RateLimitConfig struct {
	Enabled        bool          `mapstructure:"enabled"`
	RequestsPerMin int           `mapstructure:"requests_per_minute"`
	BurstSize      int           `mapstructure:"burst_size"`
	CleanupInterval time.Duration `mapstructure:"cleanup_interval"`
}

// CORSConfig holds CORS settings
type CORSConfig struct {
	Enabled        bool     `mapstructure:"enabled"`
	AllowedOrigins []string `mapstructure:"allowed_origins"`
	AllowedMethods []string `mapstructure:"allowed_methods"`
	AllowedHeaders []string `mapstructure:"allowed_headers"`
	MaxAge         int      `mapstructure:"max_age"`
}

// TLSConfig holds TLS settings
type TLSConfig struct {
	Enabled  bool   `mapstructure:"enabled"`
	CertFile string `mapstructure:"cert_file"`
	KeyFile  string `mapstructure:"key_file"`
	AutoTLS  bool   `mapstructure:"auto_tls"`
}

// LoadConfig loads configuration from various sources
func LoadConfig() (*Config, error) {
	// Set default values
	setDefaults()

	// Configure viper
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./config")
	viper.AddConfigPath("/etc/smartrag-monitoring")
	viper.AddConfigPath("$HOME/.smartrag-monitoring")

	// Environment variable support
	viper.SetEnvPrefix("SMARTRAG_MONITORING")
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))
	viper.AutomaticEnv()

	// Read config file
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	// Unmarshal into struct
	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Validate configuration
	if err := validateConfig(&config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}

// setDefaults sets default configuration values
func setDefaults() {
	// Server defaults
	viper.SetDefault("server.host", "0.0.0.0")
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.metrics_port", 9090)
	viper.SetDefault("server.read_timeout", "30s")
	viper.SetDefault("server.write_timeout", "30s")
	viper.SetDefault("server.idle_timeout", "120s")
	viper.SetDefault("server.enable_pprof", false)
	viper.SetDefault("server.enable_cors", true)

	// Monitoring defaults
	viper.SetDefault("monitoring.enabled", true)
	viper.SetDefault("monitoring.collection_interval", "30s")
	viper.SetDefault("monitoring.health_check_interval", "60s")
	viper.SetDefault("monitoring.process_monitoring", false)
	viper.SetDefault("monitoring.temperature_monitoring", false)

	// Logging defaults
	viper.SetDefault("logging.level", "info")
	viper.SetDefault("logging.format", "json")
	viper.SetDefault("logging.output", "stdout")
	viper.SetDefault("logging.max_size_mb", 100)
	viper.SetDefault("logging.max_backups", 3)
	viper.SetDefault("logging.max_age_days", 30)
	viper.SetDefault("logging.compress", true)

	// Storage defaults
	viper.SetDefault("storage.type", "memory")
	viper.SetDefault("storage.max_size_mb", 100)
	viper.SetDefault("storage.metrics_retention", "24h")
	viper.SetDefault("storage.alerts_retention", "7d")
	viper.SetDefault("storage.health_retention", "24h")
	viper.SetDefault("storage.compression_enabled", false)
	viper.SetDefault("storage.backup_enabled", false)
	viper.SetDefault("storage.backup_interval", "1h")

	// Alerting defaults
	viper.SetDefault("alerts.enabled", false)

	// Security defaults
	viper.SetDefault("security.enable_auth", false)
	viper.SetDefault("security.rate_limit.enabled", true)
	viper.SetDefault("security.rate_limit.requests_per_minute", 1000)
	viper.SetDefault("security.rate_limit.burst_size", 100)
	viper.SetDefault("security.rate_limit.cleanup_interval", "1m")
	viper.SetDefault("security.cors.enabled", true)
	viper.SetDefault("security.cors.allowed_origins", []string{"*"})
	viper.SetDefault("security.cors.allowed_methods", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"})
	viper.SetDefault("security.cors.allowed_headers", []string{"*"})
	viper.SetDefault("security.cors.max_age", 3600)
	viper.SetDefault("security.tls.enabled", false)
}

// validateConfig validates the configuration
func validateConfig(config *Config) error {
	// Validate server ports
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", config.Server.Port)
	}
	if config.Server.MetricsPort <= 0 || config.Server.MetricsPort > 65535 {
		return fmt.Errorf("invalid metrics port: %d", config.Server.MetricsPort)
	}

	// Validate monitoring intervals
	if config.Monitoring.CollectionInterval <= 0 {
		return fmt.Errorf("collection interval must be positive")
	}
	if config.Monitoring.HealthCheckInterval <= 0 {
		return fmt.Errorf("health check interval must be positive")
	}

	// Validate storage configuration
	if config.Storage.Type != "memory" && config.Storage.Type != "file" && config.Storage.Type != "postgres" {
		return fmt.Errorf("invalid storage type: %s", config.Storage.Type)
	}

	// Validate service configurations
	for i, service := range config.Monitoring.Services {
		if service.Name == "" {
			return fmt.Errorf("service %d: name is required", i)
		}
		if service.Endpoint == "" {
			return fmt.Errorf("service %s: endpoint is required", service.Name)
		}
		if service.Type != "http" && service.Type != "tcp" && service.Type != "grpc" {
			return fmt.Errorf("service %s: invalid type %s", service.Name, service.Type)
		}
	}

	// Validate alert rules
	for i, rule := range config.Alerts.Rules {
		if rule.Name == "" {
			return fmt.Errorf("alert rule %d: name is required", i)
		}
		if rule.Metric == "" {
			return fmt.Errorf("alert rule %s: metric is required", rule.Name)
		}
		if rule.Condition != "gt" && rule.Condition != "lt" && rule.Condition != "eq" && rule.Condition != "ne" {
			return fmt.Errorf("alert rule %s: invalid condition %s", rule.Name, rule.Condition)
		}
	}

	return nil
}

// ConvertToTypesConfig converts config to types.MonitoringConfig
func (c *Config) ConvertToTypesConfig() *types.MonitoringConfig {
	services := make([]types.ServiceConfig, len(c.Monitoring.Services))
	for i, s := range c.Monitoring.Services {
		services[i] = types.ServiceConfig{
			Name:           s.Name,
			Type:           s.Type,
			Endpoint:       s.Endpoint,
			Timeout:        s.Timeout,
			Interval:       s.Interval,
			Headers:        s.Headers,
			ExpectedStatus: s.ExpectedStatus,
			SSL:            s.SSL,
			Enabled:        s.Enabled,
		}
	}

	alerts := make([]types.AlertRule, len(c.Alerts.Rules))
	for i, r := range c.Alerts.Rules {
		alerts[i] = types.AlertRule{
			Name:      r.Name,
			Metric:    r.Metric,
			Condition: r.Condition,
			Threshold: r.Threshold,
			Duration:  r.Duration,
			Severity:  types.AlertSeverity(r.Severity),
			Labels:    r.Labels,
			Enabled:   r.Enabled,
		}
	}

	return &types.MonitoringConfig{
		CollectionInterval:  c.Monitoring.CollectionInterval,
		HealthCheckInterval: c.Monitoring.HealthCheckInterval,
		Services:           services,
		Alerts:            alerts,
		Storage: types.StorageConfig{
			Type:    c.Storage.Type,
			Path:    c.Storage.Path,
			MaxSize: c.Storage.MaxSizeMB,
			DSN:     c.Storage.DSN,
		},
		Retention: types.RetentionConfig{
			MetricsRetention:      c.Storage.MetricsRetention,
			AlertsRetention:       c.Storage.AlertsRetention,
			HealthChecksRetention: c.Storage.HealthRetention,
		},
	}
}