package ratelimit

import (
	"context"
	"strconv"
	"time"

	"github.com/go-redis/redis/v8"
)

type RedisRateLimiter struct {
	client      *redis.Client
	maxRequests int
	windowSize  time.Duration
}

func NewRedisRateLimiter(client *redis.Client, maxRequests int, windowSize time.Duration) *RedisRateLimiter {
	return &RedisRateLimiter{
		client:      client,
		maxRequests: maxRequests,
		windowSize:  windowSize,
	}
}

func (r *RedisRateLimiter) IsAllowed(ctx context.Context, clientID string) (allowed bool, remaining int, resetTime int64, err error) {
	now := time.Now()
	windowStart := now.Truncate(r.windowSize)
	key := "ratelimit:" + clientID + ":" + strconv.FormatInt(windowStart.Unix(), 10)

	// Lua script for atomic increment and check
	luaScript := `
		local key = KEYS[1]
		local limit = tonumber(ARGV[1])
		local ttl = tonumber(ARGV[2])
		
		local current = redis.call('GET', key)
		if current == false then
			current = 0
		else
			current = tonumber(current)
		end
		
		if current < limit then
			redis.call('INCR', key)
			redis.call('EXPIRE', key, ttl)
			return {1, limit - current - 1}
		else
			return {0, 0}
		end
	`

	result, err := r.client.Eval(ctx, luaScript, []string{key}, r.maxRequests, int(r.windowSize.Seconds())).Result()
	if err != nil {
		return false, 0, 0, err
	}

	resultSlice := result.([]interface{})
	allowed = resultSlice[0].(int64) == 1
	remaining = int(resultSlice[1].(int64))
	resetTime = windowStart.Add(r.windowSize).Unix()

	return allowed, remaining, resetTime, nil
}

func (r *RedisRateLimiter) Reset(ctx context.Context, clientID string) error {
	pattern := "ratelimit:" + clientID + ":*"
	keys, err := r.client.Keys(ctx, pattern).Result()
	if err != nil {
		return err
	}

	if len(keys) > 0 {
		return r.client.Del(ctx, keys...).Err()
	}

	return nil
}