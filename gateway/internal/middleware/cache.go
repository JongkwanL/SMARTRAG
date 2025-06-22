package middleware

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"smartrag-gateway/internal/cache"
)

type responseWriter struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

func (rw *responseWriter) Write(data []byte) (int, error) {
	rw.body.Write(data)
	return rw.ResponseWriter.Write(data)
}

func Cache(redisClient *cache.RedisClient, ttl int) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Only cache GET and POST requests
		if c.Request.Method != "GET" && c.Request.Method != "POST" {
			c.Next()
			return
		}

		// Generate cache key
		cacheKey := generateCacheKey(c)

		// Try to get from cache
		cachedResponse, err := redisClient.Get(c.Request.Context(), cacheKey)
		if err == nil && cachedResponse != "" {
			// Cache hit
			RecordCacheHit(c.FullPath())
			
			var response CachedResponse
			if err := json.Unmarshal([]byte(cachedResponse), &response); err == nil {
				// Set headers
				for key, value := range response.Headers {
					c.Header(key, value)
				}
				
				// Set status and body
				c.Data(response.StatusCode, response.ContentType, response.Body)
				c.Abort()
				return
			}
		}

		// Cache miss
		RecordCacheMiss(c.FullPath())

		// Wrap response writer to capture response
		rw := &responseWriter{
			ResponseWriter: c.Writer,
			body:          bytes.NewBuffer(nil),
		}
		c.Writer = rw

		// Process request
		c.Next()

		// Cache the response if status is 200
		if rw.Status() == http.StatusOK {
			response := CachedResponse{
				StatusCode:  rw.Status(),
				ContentType: rw.Header().Get("Content-Type"),
				Headers:     make(map[string]string),
				Body:        rw.body.Bytes(),
				Timestamp:   time.Now().Unix(),
			}

			// Copy important headers
			for _, header := range []string{"Content-Type", "Content-Encoding"} {
				if value := rw.Header().Get(header); value != "" {
					response.Headers[header] = value
				}
			}

			// Store in cache
			if responseBytes, err := json.Marshal(response); err == nil {
				redisClient.Set(c.Request.Context(), cacheKey, string(responseBytes), time.Duration(ttl)*time.Second)
			}
		}
	}
}

type CachedResponse struct {
	StatusCode  int               `json:"status_code"`
	ContentType string            `json:"content_type"`
	Headers     map[string]string `json:"headers"`
	Body        []byte            `json:"body"`
	Timestamp   int64             `json:"timestamp"`
}

func generateCacheKey(c *gin.Context) string {
	// Include method, path, query params, and body for POST requests
	key := c.Request.Method + ":" + c.Request.URL.Path

	// Add query parameters
	if c.Request.URL.RawQuery != "" {
		key += "?" + c.Request.URL.RawQuery
	}

	// Add body hash for POST requests
	if c.Request.Method == "POST" {
		if body, err := io.ReadAll(c.Request.Body); err == nil {
			// Restore body for downstream handlers
			c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
			
			// Add body hash to cache key
			hash := md5.Sum(body)
			key += ":body:" + fmt.Sprintf("%x", hash)
		}
	}

	// Add user ID if authenticated (for user-specific caching)
	if userID, exists := c.Get("user_id"); exists {
		key += ":user:" + userID.(string)
	}

	return "cache:" + key
}