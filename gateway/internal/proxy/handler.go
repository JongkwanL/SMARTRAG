package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type Handler struct {
	backendURL *url.URL
	client     *http.Client
	logger     *zap.Logger
}

func NewHandler(backendURL string, timeout time.Duration, logger *zap.Logger) (*Handler, error) {
	parsedURL, err := url.Parse(backendURL)
	if err != nil {
		return nil, err
	}

	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	return &Handler{
		backendURL: parsedURL,
		client:     client,
		logger:     logger,
	}, nil
}

func (h *Handler) ProxyRequest(c *gin.Context) {
	// Build target URL
	targetURL := &url.URL{
		Scheme: h.backendURL.Scheme,
		Host:   h.backendURL.Host,
		Path:   h.backendURL.Path + c.Request.URL.Path,
	}

	// Copy query parameters
	if c.Request.URL.RawQuery != "" {
		targetURL.RawQuery = c.Request.URL.RawQuery
	}

	// Read request body
	var bodyBytes []byte
	if c.Request.Body != nil {
		bodyBytes, _ = io.ReadAll(c.Request.Body)
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	}

	// Create new request
	req, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		targetURL.String(),
		bytes.NewBuffer(bodyBytes),
	)
	if err != nil {
		h.logger.Error("Failed to create proxy request", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "proxy request failed"})
		return
	}

	// Copy headers (excluding hop-by-hop headers)
	h.copyHeaders(req.Header, c.Request.Header)

	// Add/modify headers for proxy
	req.Header.Set("X-Forwarded-For", c.ClientIP())
	req.Header.Set("X-Forwarded-Proto", "http")
	if c.Request.TLS != nil {
		req.Header.Set("X-Forwarded-Proto", "https")
	}
	req.Header.Set("X-Forwarded-Host", c.Request.Host)

	// Execute request
	resp, err := h.client.Do(req)
	if err != nil {
		h.logger.Error("Proxy request failed", 
			zap.Error(err),
			zap.String("target_url", targetURL.String()),
		)
		c.JSON(http.StatusBadGateway, gin.H{"error": "backend unavailable"})
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	h.copyHeaders(c.Writer.Header(), resp.Header)

	// Set status code
	c.Status(resp.StatusCode)

	// Copy response body
	io.Copy(c.Writer, resp.Body)
}

func (h *Handler) copyHeaders(dst, src http.Header) {
	// Skip hop-by-hop headers
	hopByHopHeaders := map[string]bool{
		"Connection":          true,
		"Proxy-Connection":    true,
		"Keep-Alive":          true,
		"Proxy-Authenticate":  true,
		"Proxy-Authorization": true,
		"Te":                  true,
		"Trailer":             true,
		"Transfer-Encoding":   true,
		"Upgrade":             true,
	}

	for key, values := range src {
		if !hopByHopHeaders[key] {
			for _, value := range values {
				dst.Add(key, value)
			}
		}
	}
}

func (h *Handler) HealthCheck(c *gin.Context) {
	// Create health check request to backend
	healthURL := &url.URL{
		Scheme: h.backendURL.Scheme,
		Host:   h.backendURL.Host,
		Path:   "/health",
	}

	req, err := http.NewRequestWithContext(
		c.Request.Context(),
		"GET",
		healthURL.String(),
		nil,
	)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "failed to create health check request",
		})
		return
	}

	resp, err := h.client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "backend unreachable",
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"backend": "connected",
		})
	} else {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":        "unhealthy",
			"backend_status": resp.StatusCode,
		})
	}
}