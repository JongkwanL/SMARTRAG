package types

import (
	"time"
)

// SystemMetrics represents collected system metrics
type SystemMetrics struct {
	Timestamp   time.Time          `json:"timestamp"`
	CPU         CPUMetrics         `json:"cpu"`
	Memory      MemoryMetrics      `json:"memory"`
	Disk        []DiskMetrics      `json:"disk"`
	Network     []NetworkMetrics   `json:"network"`
	Process     []ProcessMetrics   `json:"process,omitempty"`
	Load        LoadMetrics        `json:"load"`
	Temperature []TemperatureMetrics `json:"temperature,omitempty"`
}

// CPUMetrics holds CPU-related metrics
type CPUMetrics struct {
	UsagePercent    float64   `json:"usage_percent"`
	CoreCount       int       `json:"core_count"`
	PerCoreUsage    []float64 `json:"per_core_usage"`
	LoadAverage1m   float64   `json:"load_average_1m"`
	LoadAverage5m   float64   `json:"load_average_5m"`
	LoadAverage15m  float64   `json:"load_average_15m"`
	UserPercent     float64   `json:"user_percent"`
	SystemPercent   float64   `json:"system_percent"`
	IdlePercent     float64   `json:"idle_percent"`
	IOWaitPercent   float64   `json:"iowait_percent"`
}

// MemoryMetrics holds memory-related metrics
type MemoryMetrics struct {
	TotalBytes     uint64  `json:"total_bytes"`
	AvailableBytes uint64  `json:"available_bytes"`
	UsedBytes      uint64  `json:"used_bytes"`
	FreeBytes      uint64  `json:"free_bytes"`
	UsagePercent   float64 `json:"usage_percent"`
	BufferedBytes  uint64  `json:"buffered_bytes"`
	CachedBytes    uint64  `json:"cached_bytes"`
	SwapTotal      uint64  `json:"swap_total"`
	SwapUsed       uint64  `json:"swap_used"`
	SwapFree       uint64  `json:"swap_free"`
	SwapPercent    float64 `json:"swap_percent"`
}

// DiskMetrics holds disk-related metrics
type DiskMetrics struct {
	Device         string  `json:"device"`
	Mountpoint     string  `json:"mountpoint"`
	Filesystem     string  `json:"filesystem"`
	TotalBytes     uint64  `json:"total_bytes"`
	UsedBytes      uint64  `json:"used_bytes"`
	FreeBytes      uint64  `json:"free_bytes"`
	UsagePercent   float64 `json:"usage_percent"`
	InodesTotal    uint64  `json:"inodes_total"`
	InodesUsed     uint64  `json:"inodes_used"`
	InodesFree     uint64  `json:"inodes_free"`
	ReadCount      uint64  `json:"read_count"`
	WriteCount     uint64  `json:"write_count"`
	ReadBytes      uint64  `json:"read_bytes"`
	WriteBytes     uint64  `json:"write_bytes"`
	ReadTime       uint64  `json:"read_time_ms"`
	WriteTime      uint64  `json:"write_time_ms"`
	IOPercent      float64 `json:"io_percent"`
}

// NetworkMetrics holds network interface metrics
type NetworkMetrics struct {
	Interface     string `json:"interface"`
	BytesSent     uint64 `json:"bytes_sent"`
	BytesReceived uint64 `json:"bytes_received"`
	PacketsSent   uint64 `json:"packets_sent"`
	PacketsRecv   uint64 `json:"packets_recv"`
	ErrorsIn      uint64 `json:"errors_in"`
	ErrorsOut     uint64 `json:"errors_out"`
	DropsIn       uint64 `json:"drops_in"`
	DropsOut      uint64 `json:"drops_out"`
	Speed         uint64 `json:"speed_mbps"`
	MTU           int    `json:"mtu"`
	IsUp          bool   `json:"is_up"`
}

// ProcessMetrics holds process-specific metrics
type ProcessMetrics struct {
	PID            int32   `json:"pid"`
	Name           string  `json:"name"`
	Username       string  `json:"username"`
	Status         string  `json:"status"`
	CPUPercent     float64 `json:"cpu_percent"`
	MemoryPercent  float64 `json:"memory_percent"`
	MemoryRSS      uint64  `json:"memory_rss"`
	MemoryVMS      uint64  `json:"memory_vms"`
	NumThreads     int32   `json:"num_threads"`
	NumFDs         int32   `json:"num_fds"`
	CreateTime     int64   `json:"create_time"`
	OpenFiles      int     `json:"open_files"`
	Connections    int     `json:"connections"`
}

// LoadMetrics holds system load metrics
type LoadMetrics struct {
	Load1  float64 `json:"load_1"`
	Load5  float64 `json:"load_5"`
	Load15 float64 `json:"load_15"`
}

// TemperatureMetrics holds temperature sensor metrics
type TemperatureMetrics struct {
	SensorKey   string  `json:"sensor_key"`
	Temperature float64 `json:"temperature_celsius"`
	High        float64 `json:"high_celsius,omitempty"`
	Critical    float64 `json:"critical_celsius,omitempty"`
}

// ServiceHealth represents health check status for a service
type ServiceHealth struct {
	ServiceName   string            `json:"service_name"`
	Status        HealthStatus      `json:"status"`
	CheckTime     time.Time         `json:"check_time"`
	ResponseTime  time.Duration     `json:"response_time"`
	Error         string            `json:"error,omitempty"`
	Details       map[string]string `json:"details,omitempty"`
	Endpoint      string            `json:"endpoint"`
	CheckType     string            `json:"check_type"`
}

// HealthStatus represents the status of a health check
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
)

// Alert represents an alert condition
type Alert struct {
	ID          string            `json:"id"`
	Title       string            `json:"title"`
	Description string            `json:"description"`
	Severity    AlertSeverity     `json:"severity"`
	Source      string            `json:"source"`
	Timestamp   time.Time         `json:"timestamp"`
	Labels      map[string]string `json:"labels"`
	Value       float64           `json:"value,omitempty"`
	Threshold   float64           `json:"threshold,omitempty"`
	Status      AlertStatus       `json:"status"`
}

// AlertSeverity represents the severity level of an alert
type AlertSeverity string

const (
	SeverityCritical AlertSeverity = "critical"
	SeverityWarning  AlertSeverity = "warning"
	SeverityInfo     AlertSeverity = "info"
)

// AlertStatus represents the current status of an alert
type AlertStatus string

const (
	AlertStatusFiring   AlertStatus = "firing"
	AlertStatusResolved AlertStatus = "resolved"
)

// MonitoringConfig represents the configuration for monitoring
type MonitoringConfig struct {
	CollectionInterval time.Duration         `json:"collection_interval"`
	HealthCheckInterval time.Duration         `json:"health_check_interval"`
	Services           []ServiceConfig       `json:"services"`
	Alerts             []AlertRule           `json:"alerts"`
	Storage            StorageConfig         `json:"storage"`
	Retention          RetentionConfig       `json:"retention"`
}

// ServiceConfig defines a service to monitor
type ServiceConfig struct {
	Name        string            `json:"name"`
	Type        string            `json:"type"` // http, tcp, grpc
	Endpoint    string            `json:"endpoint"`
	Timeout     time.Duration     `json:"timeout"`
	Interval    time.Duration     `json:"interval"`
	Headers     map[string]string `json:"headers,omitempty"`
	ExpectedStatus int            `json:"expected_status,omitempty"`
	SSL         bool              `json:"ssl,omitempty"`
	Enabled     bool              `json:"enabled"`
}

// AlertRule defines conditions for triggering alerts
type AlertRule struct {
	Name        string            `json:"name"`
	Metric      string            `json:"metric"`
	Condition   string            `json:"condition"` // gt, lt, eq, ne
	Threshold   float64           `json:"threshold"`
	Duration    time.Duration     `json:"duration"`
	Severity    AlertSeverity     `json:"severity"`
	Labels      map[string]string `json:"labels"`
	Enabled     bool              `json:"enabled"`
}

// StorageConfig defines storage options for metrics
type StorageConfig struct {
	Type     string `json:"type"` // memory, file, postgres
	Path     string `json:"path,omitempty"`
	MaxSize  int64  `json:"max_size_mb,omitempty"`
	DSN      string `json:"dsn,omitempty"`
}

// RetentionConfig defines data retention policies
type RetentionConfig struct {
	MetricsRetention     time.Duration `json:"metrics_retention"`
	AlertsRetention      time.Duration `json:"alerts_retention"`
	HealthChecksRetention time.Duration `json:"health_checks_retention"`
}

// MonitoringStats represents monitoring agent statistics
type MonitoringStats struct {
	StartTime           time.Time `json:"start_time"`
	UptimeSeconds       int64     `json:"uptime_seconds"`
	MetricsCollected    int64     `json:"metrics_collected"`
	HealthChecksRun     int64     `json:"health_checks_run"`
	AlertsFired         int64     `json:"alerts_fired"`
	AlertsResolved      int64     `json:"alerts_resolved"`
	StorageSize         int64     `json:"storage_size_bytes"`
	LastCollectionTime  time.Time `json:"last_collection_time"`
	CollectionErrors    int64     `json:"collection_errors"`
	HealthCheckErrors   int64     `json:"health_check_errors"`
}

// MetricPoint represents a single metric data point
type MetricPoint struct {
	Name      string            `json:"name"`
	Value     float64           `json:"value"`
	Timestamp time.Time         `json:"timestamp"`
	Labels    map[string]string `json:"labels"`
	Unit      string            `json:"unit,omitempty"`
}