package middleware

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	activeConnections = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "gateway_active_connections",
			Help: "Number of active connections",
		},
	)

	cacheHits = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_cache_hits_total",
			Help: "Total number of cache hits",
		},
		[]string{"endpoint"},
	)

	cacheMisses = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_cache_misses_total",
			Help: "Total number of cache misses",
		},
		[]string{"endpoint"},
	)
)

func init() {
	prometheus.MustRegister(
		requestsTotal,
		requestDuration,
		activeConnections,
		cacheHits,
		cacheMisses,
	)
}

func Metrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		activeConnections.Inc()

		// Process request
		c.Next()

		// Record metrics
		duration := time.Since(start).Seconds()
		status := strconv.Itoa(c.Writer.Status())
		endpoint := c.FullPath()
		method := c.Request.Method

		requestsTotal.WithLabelValues(method, endpoint, status).Inc()
		requestDuration.WithLabelValues(method, endpoint).Observe(duration)
		activeConnections.Dec()
	}
}

func PrometheusHandler() *promhttp.Handler {
	return promhttp.Handler().(*promhttp.Handler)
}

// Helper functions for cache metrics
func RecordCacheHit(endpoint string) {
	cacheHits.WithLabelValues(endpoint).Inc()
}

func RecordCacheMiss(endpoint string) {
	cacheMisses.WithLabelValues(endpoint).Inc()
}