package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all prometheus metrics for the vector search service
type Metrics struct {
	// Request counters
	searchRequests *prometheus.CounterVec
	addRequests    *prometheus.CounterVec
	deleteRequests *prometheus.CounterVec

	// Error counters
	searchErrors *prometheus.CounterVec
	addErrors    *prometheus.CounterVec
	deleteErrors *prometheus.CounterVec

	// Latency histograms
	searchLatency *prometheus.HistogramVec
	addLatency    *prometheus.HistogramVec
	deleteLatency *prometheus.HistogramVec

	// Index metrics
	totalVectors    prometheus.Gauge
	indexSizeMB     prometheus.Gauge
	searchResults   prometheus.Histogram
	vectorsAdded    prometheus.Counter
	vectorsDeleted  prometheus.Counter

	// System metrics
	memoryUsage prometheus.Gauge
	cpuUsage    prometheus.Gauge
}

// NewMetrics creates a new metrics instance with all prometheus collectors
func NewMetrics() *Metrics {
	return &Metrics{
		// Request counters
		searchRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_search_requests_total",
				Help: "Total number of search requests",
			},
			[]string{"status"},
		),
		addRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_add_requests_total",
				Help: "Total number of add vector requests",
			},
			[]string{"status"},
		),
		deleteRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_delete_requests_total",
				Help: "Total number of delete vector requests",
			},
			[]string{"status"},
		),

		// Error counters
		searchErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_search_errors_total",
				Help: "Total number of search errors",
			},
			[]string{"error_type"},
		),
		addErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_add_errors_total",
				Help: "Total number of add vector errors",
			},
			[]string{"error_type"},
		),
		deleteErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vector_delete_errors_total",
				Help: "Total number of delete vector errors",
			},
			[]string{"error_type"},
		),

		// Latency histograms
		searchLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vector_search_duration_seconds",
				Help:    "Duration of search operations",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~32s
			},
			[]string{"operation"},
		),
		addLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vector_add_duration_seconds",
				Help:    "Duration of add vector operations",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"operation"},
		),
		deleteLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vector_delete_duration_seconds",
				Help:    "Duration of delete vector operations",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"operation"},
		),

		// Index metrics
		totalVectors: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "vector_index_total_vectors",
				Help: "Total number of vectors in the index",
			},
		),
		indexSizeMB: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "vector_index_size_mb",
				Help: "Size of the vector index in megabytes",
			},
		),
		searchResults: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "vector_search_results_count",
				Help:    "Number of results returned by search operations",
				Buckets: prometheus.LinearBuckets(0, 10, 20), // 0 to 200 results
			},
		),
		vectorsAdded: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "vector_vectors_added_total",
				Help: "Total number of vectors added to the index",
			},
		),
		vectorsDeleted: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "vector_vectors_deleted_total",
				Help: "Total number of vectors deleted from the index",
			},
		),

		// System metrics
		memoryUsage: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "vector_memory_usage_bytes",
				Help: "Memory usage of the vector search service",
			},
		),
		cpuUsage: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "vector_cpu_usage_percent",
				Help: "CPU usage percentage of the vector search service",
			},
		),
	}
}

// Request metrics
func (m *Metrics) IncSearchRequests() {
	m.searchRequests.WithLabelValues("success").Inc()
}

func (m *Metrics) IncAddRequests() {
	m.addRequests.WithLabelValues("success").Inc()
}

func (m *Metrics) IncDeleteRequests() {
	m.deleteRequests.WithLabelValues("success").Inc()
}

// Error metrics
func (m *Metrics) IncSearchErrors(errorType string) {
	m.searchErrors.WithLabelValues(errorType).Inc()
}

func (m *Metrics) IncAddErrors(errorType string) {
	m.addErrors.WithLabelValues(errorType).Inc()
}

func (m *Metrics) IncDeleteErrors(errorType string) {
	m.deleteErrors.WithLabelValues(errorType).Inc()
}

// Latency metrics
func (m *Metrics) RecordSearchLatency(duration time.Duration) {
	m.searchLatency.WithLabelValues("search").Observe(duration.Seconds())
}

func (m *Metrics) RecordAddLatency(duration time.Duration) {
	m.addLatency.WithLabelValues("add").Observe(duration.Seconds())
}

func (m *Metrics) RecordDeleteLatency(duration time.Duration) {
	m.deleteLatency.WithLabelValues("delete").Observe(duration.Seconds())
}

// Index metrics
func (m *Metrics) UpdateTotalVectors(count int64) {
	m.totalVectors.Set(float64(count))
}

func (m *Metrics) UpdateIndexSize(sizeMB float64) {
	m.indexSizeMB.Set(sizeMB)
}

func (m *Metrics) RecordSearchResults(count int) {
	m.searchResults.Observe(float64(count))
}

func (m *Metrics) RecordVectorsAdded(count int) {
	m.vectorsAdded.Add(float64(count))
}

func (m *Metrics) RecordVectorsDeleted(count int) {
	m.vectorsDeleted.Add(float64(count))
}

// System metrics
func (m *Metrics) UpdateMemoryUsage(bytes uint64) {
	m.memoryUsage.Set(float64(bytes))
}

func (m *Metrics) UpdateCPUUsage(percent float64) {
	m.cpuUsage.Set(percent)
}

// GetHandler returns the prometheus metrics HTTP handler
func (m *Metrics) GetHandler() http.Handler {
	return promhttp.Handler()
}

// StartMetricsServer starts a dedicated HTTP server for metrics
func (m *Metrics) StartMetricsServer(addr string) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", m.GetHandler())
	
	// Add health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
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