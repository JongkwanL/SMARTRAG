package config

import (
	"os"
	"strconv"
)

type Config struct {
	Server ServerConfig `yaml:"server"`
	Vector VectorConfig `yaml:"vector"`
	Metrics MetricsConfig `yaml:"metrics"`
}

type ServerConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	Workers int    `yaml:"workers"`
}

type VectorConfig struct {
	Dimension     int     `yaml:"dimension"`
	IndexType     string  `yaml:"index_type"` // "flat", "ivf", "hnsw"
	MetricType    string  `yaml:"metric_type"` // "l2", "ip", "cosine"
	IndexPath     string  `yaml:"index_path"`
	MaxVectors    int64   `yaml:"max_vectors"`
	EfConstruction int    `yaml:"ef_construction"` // for HNSW
	M             int     `yaml:"m"`               // for HNSW
	NList         int     `yaml:"nlist"`           // for IVF
}

type MetricsConfig struct {
	Enabled bool   `yaml:"enabled"`
	Port    int    `yaml:"port"`
	Path    string `yaml:"path"`
}

func Load() *Config {
	return &Config{
		Server: ServerConfig{
			Host:    getEnv("SERVER_HOST", "0.0.0.0"),
			Port:    getEnvInt("SERVER_PORT", 50051),
			Workers: getEnvInt("SERVER_WORKERS", 4),
		},
		Vector: VectorConfig{
			Dimension:      getEnvInt("VECTOR_DIMENSION", 384),
			IndexType:      getEnv("VECTOR_INDEX_TYPE", "hnsw"),
			MetricType:     getEnv("VECTOR_METRIC_TYPE", "cosine"),
			IndexPath:      getEnv("VECTOR_INDEX_PATH", "./data/vector.index"),
			MaxVectors:     int64(getEnvInt("VECTOR_MAX_VECTORS", 1000000)),
			EfConstruction: getEnvInt("VECTOR_EF_CONSTRUCTION", 200),
			M:              getEnvInt("VECTOR_M", 16),
			NList:          getEnvInt("VECTOR_NLIST", 1024),
		},
		Metrics: MetricsConfig{
			Enabled: getEnvBool("METRICS_ENABLED", true),
			Port:    getEnvInt("METRICS_PORT", 9090),
			Path:    getEnv("METRICS_PATH", "/metrics"),
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

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}