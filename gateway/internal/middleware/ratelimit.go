package middleware

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"smartrag-gateway/internal/ratelimit"
)

func RateLimit(rateLimiter *ratelimit.RedisRateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Get client identifier (IP or user ID)
		clientID := getClientID(c)

		// Check rate limit
		allowed, remaining, resetTime, err := rateLimiter.IsAllowed(c.Request.Context(), clientID)
		if err != nil {
			// Log error but don't block request
			c.Header("X-RateLimit-Error", err.Error())
			c.Next()
			return
		}

		// Set rate limit headers
		c.Header("X-RateLimit-Remaining", strconv.Itoa(remaining))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(resetTime, 10))

		if !allowed {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":     "rate limit exceeded",
				"remaining": remaining,
				"reset":     resetTime,
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

func getClientID(c *gin.Context) string {
	// Try to get user ID from context (if authenticated)
	if userID, exists := c.Get("user_id"); exists {
		return "user:" + userID.(string)
	}

	// Fall back to IP address
	return "ip:" + c.ClientIP()
}