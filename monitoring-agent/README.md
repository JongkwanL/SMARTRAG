# SmartRAG Monitoring Agent

A lightweight, high-performance monitoring agent written in Go for the SmartRAG ecosystem. Provides comprehensive system metrics collection, service health monitoring, and Prometheus integration.

## Features

- **System Metrics Collection**: CPU, memory, disk, network, load average, and temperature monitoring
- **Service Health Monitoring**: HTTP, TCP, and gRPC health checks with configurable intervals
- **Process Monitoring**: Optional detailed process metrics with filtering capabilities
- **Prometheus Integration**: Native Prometheus metrics export with comprehensive labels
- **RESTful API**: Complete REST API for metrics, health checks, and agent management
- **Configurable Alerting**: Rule-based alerting with multiple notification channels
- **Lightweight**: Minimal resource footprint with high performance
- **Docker Support**: Full containerization with health checks
- **Security**: Authentication, rate limiting, CORS, and TLS support

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │◄───│  Monitoring     │────►│   SmartRAG      │
│   Scraping      │    │    Agent        │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   System        │
                       │   Resources     │
                       └─────────────────┘
```

## Quick Start

### Using Docker

1. **Pull and run the container**:
```bash
docker run -d \
  --name smartrag-monitoring \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /proc:/host/proc:ro \
  -v /sys:/host/sys:ro \
  smartrag-monitoring-agent:latest
```

2. **Check health**:
```bash
curl http://localhost:8080/health
```

3. **View metrics**:
```bash
curl http://localhost:9090/metrics
```

### Building from Source

1. **Build the application**:
```bash
go build -o monitoring-agent ./cmd
```

2. **Run with default configuration**:
```bash
./monitoring-agent
```

3. **Run with custom configuration**:
```bash
cp config.example.yaml config.yaml
# Edit config.yaml as needed
./monitoring-agent
```

## Configuration

The monitoring agent uses YAML configuration with environment variable overrides. See [config.example.yaml](config.example.yaml) for a complete example.

### Environment Variables

All configuration options can be overridden using environment variables with the prefix `SMARTRAG_MONITORING_`:

```bash
# Server configuration
export SMARTRAG_MONITORING_SERVER_HOST=0.0.0.0
export SMARTRAG_MONITORING_SERVER_PORT=8080
export SMARTRAG_MONITORING_METRICS_PORT=9090

# Monitoring configuration
export SMARTRAG_MONITORING_MONITORING_ENABLED=true
export SMARTRAG_MONITORING_MONITORING_COLLECTION_INTERVAL=30s
export SMARTRAG_MONITORING_MONITORING_HEALTH_CHECK_INTERVAL=60s

# Logging configuration
export SMARTRAG_MONITORING_LOGGING_LEVEL=info
export SMARTRAG_MONITORING_LOGGING_FORMAT=json
```

### Key Configuration Sections

#### System Monitoring
```yaml
monitoring:
  enabled: true
  collection_interval: 30s
  process_monitoring: true
  temperature_monitoring: false
  network_interfaces: ["eth0", "wlan0"]
  disk_mountpoints: ["/", "/data"]
```

#### Service Health Checks
```yaml
monitoring:
  services:
    - name: "smartrag-api"
      type: "http"
      endpoint: "http://localhost:8000/health"
      timeout: 10s
      interval: 60s
      expected_status: 200
      enabled: true
```

#### Alerting Rules
```yaml
alerts:
  enabled: true
  rules:
    - name: "high_cpu_usage"
      metric: "cpu_usage_percent"
      condition: "gt"
      threshold: 90.0
      duration: 5m
      severity: "critical"
```

## API Endpoints

### Health and Status
- `GET /health` - Basic health check
- `GET /ready` - Readiness check for Kubernetes
- `GET /status` - Detailed status information

### Metrics
- `GET /metrics/system` - Current system metrics
- `GET /metrics/health` - Current health check results
- `GET /api/v1/metrics` - All metrics (system + health)

### Management
- `GET /api/v1/stats` - Agent statistics
- `GET /api/v1/config` - Current configuration (sanitized)
- `POST /api/v1/health/check` - Trigger immediate health check

### Prometheus Metrics
- `GET :9090/metrics` - Prometheus format metrics

## Metrics Collection

### System Metrics

**CPU Metrics**:
- `system_cpu_usage_percent` - Overall CPU usage percentage
- `system_load_average{duration}` - Load average (1m, 5m, 15m)

**Memory Metrics**:
- `system_memory_usage_percent` - Memory usage percentage
- `system_memory_usage_bytes` - Memory usage in bytes
- `system_memory_total_bytes` - Total system memory

**Disk Metrics**:
- `system_disk_usage_percent{device,mountpoint,filesystem}` - Disk usage percentage
- `system_disk_usage_bytes{device,mountpoint,filesystem}` - Disk usage in bytes
- `system_disk_total_bytes{device,mountpoint,filesystem}` - Total disk space

**Network Metrics**:
- `system_network_bytes_total{interface,direction}` - Network bytes transferred
- `system_network_packets_total{interface,direction}` - Network packets transferred
- `system_network_errors_total{interface,direction}` - Network errors

**Process Metrics** (optional):
- `process_cpu_usage_percent{name,pid,username}` - Process CPU usage
- `process_memory_usage_percent{name,pid,username}` - Process memory usage
- `process_memory_usage_bytes{name,pid,username,type}` - Process memory (RSS/VMS)

### Health Check Metrics

- `service_health_status{service,endpoint,type}` - Service health status (0-3)
- `service_response_time_seconds{service,endpoint,type,status}` - Response time histogram
- `health_checks_total{service,status}` - Total health checks counter

### Agent Metrics

- `monitoring_agent_uptime_seconds` - Agent uptime
- `monitoring_agent_metrics_collected_total` - Total metrics collected
- `monitoring_agent_collection_duration_seconds` - Collection duration histogram
- `monitoring_agent_collection_errors_total{type}` - Collection errors

## Health Check Types

### HTTP Health Checks
```yaml
- name: "web-service"
  type: "http"
  endpoint: "https://api.example.com/health"
  method: "GET"
  headers:
    Authorization: "Bearer token"
  expected_status: 200
  timeout: 10s
  ssl: true
```

### TCP Health Checks
```yaml
- name: "database"
  type: "tcp"
  endpoint: "localhost:5432"
  timeout: 5s
```

### gRPC Health Checks
```yaml
- name: "grpc-service"
  type: "grpc"
  endpoint: "localhost:9000"
  ssl: false
  timeout: 10s
```

## Alerting

The monitoring agent supports rule-based alerting with multiple notification channels:

### Alert Rules
```yaml
alerts:
  rules:
    - name: "high_cpu_usage"
      metric: "cpu_usage_percent"
      condition: "gt"  # gt, lt, eq, ne
      threshold: 90.0
      duration: 5m
      severity: "critical"  # critical, warning, info
      enabled: true
```

### Notification Channels

**Webhooks**:
```yaml
alerts:
  webhooks:
    - name: "slack_webhook"
      url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
      method: "POST"
      timeout: 10s
```

**Email (SMTP)**:
```yaml
alerts:
  smtp:
    host: "smtp.example.com"
    port: 587
    username: "alerts@example.com"
    password: "password"
    from: "monitoring@smartrag.local"
    to: ["admin@example.com"]
    use_tls: true
```

## Security

### Authentication
```yaml
security:
  enable_auth: true
  api_keys: ["your-api-key"]
  
  basic_auth:
    username: "admin"
    password: "secure-password"
```

### Rate Limiting
```yaml
security:
  rate_limit:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
```

### TLS
```yaml
security:
  tls:
    enabled: true
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
```

### IP Filtering
```yaml
security:
  allowed_ips: ["192.168.1.0/24", "10.0.0.1"]
  blocked_ips: ["192.168.1.100"]
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  monitoring-agent:
    image: smartrag-monitoring-agent:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - ./config.yaml:/root/config.yaml
    environment:
      - SMARTRAG_MONITORING_LOGGING_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: monitoring-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: monitoring-agent
  template:
    metadata:
      labels:
        app: monitoring-agent
    spec:
      containers:
      - name: monitoring-agent
        image: smartrag-monitoring-agent:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: SMARTRAG_MONITORING_LOGGING_LEVEL
          value: "info"
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
```

### Prometheus Integration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'smartrag-monitoring'
    static_configs:
      - targets: ['monitoring-agent:9090']
    scrape_interval: 30s
    metrics_path: /metrics
```

## Performance

### Resource Usage
- **Memory**: ~10-50MB depending on monitoring scope
- **CPU**: <1% on modern systems during collection
- **Network**: Minimal, only health check traffic

### Scalability
- **Concurrent Health Checks**: Unlimited (configurable timeout)
- **Metrics Collection**: Sub-second collection times
- **API Throughput**: 1000+ requests/second with rate limiting

### Optimization Tips

1. **Disable unnecessary monitoring**:
```yaml
monitoring:
  process_monitoring: false
  temperature_monitoring: false
```

2. **Adjust collection intervals**:
```yaml
monitoring:
  collection_interval: 60s  # Reduce frequency
  health_check_interval: 300s
```

3. **Filter network interfaces and disks**:
```yaml
monitoring:
  network_interfaces: ["eth0"]  # Only monitor specific interfaces
  disk_mountpoints: ["/"]       # Only monitor specific mounts
```

## Troubleshooting

### Common Issues

**High Memory Usage**:
- Disable process monitoring if not needed
- Reduce collection frequency
- Limit network interfaces and disk monitoring

**Health Check Timeouts**:
- Increase timeout values in service configuration
- Check network connectivity to monitored services
- Verify service endpoints are correct

**Missing Metrics**:
- Check if monitoring is enabled in configuration
- Verify system permissions for accessing /proc and /sys
- Review logs for collection errors

### Debug Mode

Enable debug logging:
```bash
export SMARTRAG_MONITORING_LOGGING_LEVEL=debug
./monitoring-agent
```

Or via configuration:
```yaml
logging:
  level: debug
```

### Log Analysis

The agent provides structured JSON logs:
```json
{
  "level": "info",
  "ts": "2024-01-01T00:00:00.000Z",
  "msg": "System metrics collected",
  "duration": "150ms",
  "cpu_percent": 45.2,
  "memory_percent": 67.8
}
```

## Development

### Building

```bash
# Install dependencies
go mod download

# Build
go build -o monitoring-agent ./cmd

# Build with version info
go build -ldflags "-X main.version=v1.0.0 -X main.commit=$(git rev-parse --short HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" -o monitoring-agent ./cmd
```

### Testing

```bash
# Run tests
go test ./...

# Run with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Docker Build

```bash
# Build image
docker build -t smartrag-monitoring-agent .

# Run locally
docker run -p 8080:8080 -p 9090:9090 smartrag-monitoring-agent
```

## Integration with SmartRAG

This monitoring agent is designed to work seamlessly with the SmartRAG ecosystem:

1. **Service Discovery**: Automatically monitors all SmartRAG components
2. **Health Checks**: Validates connectivity to API Gateway, Vector Search, Streaming Proxy
3. **Resource Monitoring**: Tracks system resources used by SmartRAG services
4. **Alerting**: Notifies on service failures or resource exhaustion
5. **Metrics Export**: Provides metrics to Prometheus for visualization in Grafana

The agent acts as a centralized monitoring point for the entire SmartRAG deployment, providing visibility into system health and performance.