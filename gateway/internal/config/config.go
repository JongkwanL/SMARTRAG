package config

import (
	"os"
	"strconv"
	"strings"
)

type Config struct {
	Environment string
	Server      ServerConfig
	Backend     BackendConfig
	Redis       RedisConfig
	Auth        AuthConfig
	RateLimit   RateLimitConfig
	CORS        CORSConfig
}

type ServerConfig struct {
	Port         int
	ReadTimeout  int
	WriteTimeout int
	IdleTimeout  int
}

type BackendConfig struct {
	URL             string
	MaxConns        int
	MaxIdleConns    int
	ConnTimeout     int
	ResponseTimeout int
	MaxRetries      int
}

type RedisConfig struct {
	Address  string
	Password string
	DB       int
	PoolSize int
}

type AuthConfig struct {
	SecretKey string
	TokenTTL  int
}

type RateLimitConfig struct {
	RequestsPerMinute int
	BurstSize         int
	WindowSize        int
}

type CORSConfig struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
}

func Load() *Config {
	return &Config{
		Environment: getEnv("ENVIRONMENT", "development"),
		Server: ServerConfig{
			Port:         getEnvInt("PORT", 8080),
			ReadTimeout:  getEnvInt("READ_TIMEOUT", 30),
			WriteTimeout: getEnvInt("WRITE_TIMEOUT", 30),
			IdleTimeout:  getEnvInt("IDLE_TIMEOUT", 120),
		},
		Backend: BackendConfig{
			URL:             getEnv("BACKEND_URL", "http://localhost:8000"),
			MaxConns:        getEnvInt("BACKEND_MAX_CONNS", 100),
			MaxIdleConns:    getEnvInt("BACKEND_MAX_IDLE_CONNS", 10),
			ConnTimeout:     getEnvInt("BACKEND_CONN_TIMEOUT", 10),
			ResponseTimeout: getEnvInt("BACKEND_RESPONSE_TIMEOUT", 30),
			MaxRetries:      getEnvInt("BACKEND_MAX_RETRIES", 3),
		},
		Redis: RedisConfig{
			Address:  getEnv("REDIS_ADDRESS", "localhost:6379"),
			Password: getEnv("REDIS_PASSWORD", ""),
			DB:       getEnvInt("REDIS_DB", 0),
			PoolSize: getEnvInt("REDIS_POOL_SIZE", 10),
		},
		Auth: AuthConfig{
			SecretKey: getEnv("JWT_SECRET_KEY", "your-secret-key"),
			TokenTTL:  getEnvInt("JWT_TOKEN_TTL", 3600),
		},
		RateLimit: RateLimitConfig{
			RequestsPerMinute: getEnvInt("RATE_LIMIT_RPM", 100),
			BurstSize:         getEnvInt("RATE_LIMIT_BURST", 20),
			WindowSize:        getEnvInt("RATE_LIMIT_WINDOW", 60),
		},
		CORS: CORSConfig{
			AllowedOrigins: getEnvSlice("CORS_ALLOWED_ORIGINS", []string{"*"}),
			AllowedMethods: getEnvSlice("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
			AllowedHeaders: getEnvSlice("CORS_ALLOWED_HEADERS", []string{"*"}),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		return strings.Split(value, ",")
	}
	return defaultValue
}