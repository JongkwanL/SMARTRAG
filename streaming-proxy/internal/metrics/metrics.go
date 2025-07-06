package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all prometheus metrics for the streaming proxy
type Metrics struct {
	// Connection metrics
	activeConnections *prometheus.GaugeVec
	totalConnections  *prometheus.CounterVec
	connectionDuration *prometheus.HistogramVec

	// Message metrics
	messagesProcessed *prometheus.CounterVec
	bytesTransferred  *prometheus.CounterVec
	
	// Streaming metrics
	streamRequestsTotal    *prometheus.CounterVec
	streamDuration        *prometheus.HistogramVec
	activeStreams         prometheus.Gauge
	
	// Backend metrics
	backendRequestsTotal   *prometheus.CounterVec
	backendRequestDuration *prometheus.HistogramVec
	backendConnectionPool  prometheus.Gauge
	
	// Error metrics
	errorsTotal *prometheus.CounterVec
	
	// System metrics
	memoryUsage prometheus.Gauge
	cpuUsage    prometheus.Gauge
	goroutines  prometheus.Gauge
}

// NewMetrics creates a new metrics instance
func NewMetrics() *Metrics {
	return &Metrics{
		// Connection metrics
		activeConnections: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_active_connections",
				Help: "Number of active client connections",
			},
			[]string{"protocol"}, // websocket, sse
		),
		totalConnections: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_connections_total",
				Help: "Total number of client connections",
			},
			[]string{"protocol", "status"}, // established, closed, error
		),
		connectionDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "streaming_proxy_connection_duration_seconds",
				Help:    "Duration of client connections",
				Buckets: prometheus.ExponentialBuckets(1, 2, 15), // 1s to ~9h
			},
			[]string{"protocol"},
		),

		// Message metrics
		messagesProcessed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_messages_processed_total",
				Help: "Total number of messages processed",
			},
			[]string{"protocol", "type"}, // chunk, metadata, error, etc.
		),
		bytesTransferred: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_bytes_transferred_total",
				Help: "Total bytes transferred to clients",
			},
			[]string{"protocol", "direction"}, // sent, received
		),

		// Streaming metrics
		streamRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_stream_requests_total",
				Help: "Total number of streaming requests",
			},
			[]string{"status"}, // success, error
		),
		streamDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "streaming_proxy_stream_duration_seconds",
				Help:    "Duration of streaming requests",
				Buckets: prometheus.ExponentialBuckets(0.1, 2, 15), // 100ms to ~55min
			},
			[]string{"status"},
		),
		activeStreams: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_active_streams",
				Help: "Number of active backend streams",
			},
		),

		// Backend metrics
		backendRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_backend_requests_total",
				Help: "Total number of requests to backend",
			},
			[]string{"method", "status_code"},
		),
		backendRequestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "streaming_proxy_backend_request_duration_seconds",
				Help:    "Duration of requests to backend",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method"},
		),
		backendConnectionPool: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_backend_connection_pool",
				Help: "Number of connections in backend connection pool",
			},
		),

		// Error metrics
		errorsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streaming_proxy_errors_total",
				Help: "Total number of errors",
			},
			[]string{"type", "component"}, // connection, stream, backend
		),

		// System metrics
		memoryUsage: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_memory_usage_bytes",
				Help: "Memory usage in bytes",
			},
		),
		cpuUsage: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_cpu_usage_percent",
				Help: "CPU usage percentage",
			},
		),
		goroutines: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "streaming_proxy_goroutines",
				Help: "Number of goroutines",
			},
		),
	}
}

// Connection Metrics

func (m *Metrics) IncActiveConnections(protocol string) {
	m.activeConnections.WithLabelValues(protocol).Inc()
}

func (m *Metrics) DecActiveConnections(protocol string) {
	m.activeConnections.WithLabelValues(protocol).Dec()
}

func (m *Metrics) IncTotalConnections(protocol, status string) {
	m.totalConnections.WithLabelValues(protocol, status).Inc()
}

func (m *Metrics) ObserveConnectionDuration(protocol string, duration time.Duration) {
	m.connectionDuration.WithLabelValues(protocol).Observe(duration.Seconds())
}

// Message Metrics

func (m *Metrics) IncMessagesProcessed(protocol, messageType string) {
	m.messagesProcessed.WithLabelValues(protocol, messageType).Inc()
}

func (m *Metrics) AddBytesTransferred(protocol, direction string, bytes int64) {
	m.bytesTransferred.WithLabelValues(protocol, direction).Add(float64(bytes))
}

// Streaming Metrics

func (m *Metrics) IncStreamRequests(status string) {
	m.streamRequestsTotal.WithLabelValues(status).Inc()
}

func (m *Metrics) ObserveStreamDuration(status string, duration time.Duration) {
	m.streamDuration.WithLabelValues(status).Observe(duration.Seconds())
}

func (m *Metrics) IncActiveStreams() {
	m.activeStreams.Inc()
}

func (m *Metrics) DecActiveStreams() {
	m.activeStreams.Dec()
}

// Backend Metrics

func (m *Metrics) IncBackendRequests(method, statusCode string) {
	m.backendRequestsTotal.WithLabelValues(method, statusCode).Inc()
}

func (m *Metrics) ObserveBackendRequestDuration(method string, duration time.Duration) {
	m.backendRequestDuration.WithLabelValues(method).Observe(duration.Seconds())
}

func (m *Metrics) SetBackendConnectionPool(count float64) {
	m.backendConnectionPool.Set(count)
}

// Error Metrics

func (m *Metrics) IncErrors(errorType, component string) {
	m.errorsTotal.WithLabelValues(errorType, component).Inc()
}

// System Metrics

func (m *Metrics) SetMemoryUsage(bytes uint64) {
	m.memoryUsage.Set(float64(bytes))
}

func (m *Metrics) SetCPUUsage(percent float64) {
	m.cpuUsage.Set(percent)
}

func (m *Metrics) SetGoroutines(count int) {
	m.goroutines.Set(float64(count))
}

// GetHandler returns the prometheus metrics HTTP handler
func (m *Metrics) GetHandler() http.Handler {
	return promhttp.Handler()
}

// StartMetricsServer starts the metrics HTTP server
func (m *Metrics) StartMetricsServer(addr string) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", m.GetHandler())
	
	// Add health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Add readiness check endpoint
	mux.HandleFunc("/ready", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("READY"))
	})

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  15 * time.Second,
	}

	return server.ListenAndServe()
}