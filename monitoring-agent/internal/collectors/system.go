package collectors

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/load"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
	"github.com/shirou/gopsutil/v3/process"
	"go.uber.org/zap"

	"smartrag-monitoring-agent/pkg/types"
)

// SystemCollector collects system metrics using gopsutil
type SystemCollector struct {
	logger              *zap.Logger
	processMonitoring   bool
	processFilters      []string
	networkInterfaces   []string
	diskMountpoints     []string
	temperatureEnabled  bool
	lastNetworkStats    map[string]net.IOCountersStat
	lastDiskStats       map[string]disk.IOCountersStat
	mu                  sync.RWMutex
}

// NewSystemCollector creates a new system metrics collector
func NewSystemCollector(logger *zap.Logger, processMonitoring bool, processFilters []string, networkInterfaces []string, diskMountpoints []string, temperatureEnabled bool) *SystemCollector {
	return &SystemCollector{
		logger:              logger,
		processMonitoring:   processMonitoring,
		processFilters:      processFilters,
		networkInterfaces:   networkInterfaces,
		diskMountpoints:     diskMountpoints,
		temperatureEnabled:  temperatureEnabled,
		lastNetworkStats:    make(map[string]net.IOCountersStat),
		lastDiskStats:       make(map[string]disk.IOCountersStat),
	}
}

// Collect gathers all system metrics
func (c *SystemCollector) Collect(ctx context.Context) (*types.SystemMetrics, error) {
	start := time.Now()
	defer func() {
		c.logger.Debug("System metrics collection completed",
			zap.Duration("duration", time.Since(start)),
		)
	}()

	metrics := &types.SystemMetrics{
		Timestamp: time.Now(),
	}

	// Collect metrics concurrently
	var wg sync.WaitGroup
	errCh := make(chan error, 6)

	// CPU metrics
	wg.Add(1)
	go func() {
		defer wg.Done()
		if cpuMetrics, err := c.collectCPUMetrics(ctx); err != nil {
			errCh <- fmt.Errorf("cpu metrics: %w", err)
		} else {
			metrics.CPU = cpuMetrics
		}
	}()

	// Memory metrics
	wg.Add(1)
	go func() {
		defer wg.Done()
		if memMetrics, err := c.collectMemoryMetrics(ctx); err != nil {
			errCh <- fmt.Errorf("memory metrics: %w", err)
		} else {
			metrics.Memory = memMetrics
		}
	}()

	// Disk metrics
	wg.Add(1)
	go func() {
		defer wg.Done()
		if diskMetrics, err := c.collectDiskMetrics(ctx); err != nil {
			errCh <- fmt.Errorf("disk metrics: %w", err)
		} else {
			metrics.Disk = diskMetrics
		}
	}()

	// Network metrics
	wg.Add(1)
	go func() {
		defer wg.Done()
		if netMetrics, err := c.collectNetworkMetrics(ctx); err != nil {
			errCh <- fmt.Errorf("network metrics: %w", err)
		} else {
			metrics.Network = netMetrics
		}
	}()

	// Load metrics
	wg.Add(1)
	go func() {
		defer wg.Done()
		if loadMetrics, err := c.collectLoadMetrics(ctx); err != nil {
			errCh <- fmt.Errorf("load metrics: %w", err)
		} else {
			metrics.Load = loadMetrics
		}
	}()

	// Process metrics (optional)
	if c.processMonitoring {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if processMetrics, err := c.collectProcessMetrics(ctx); err != nil {
				errCh <- fmt.Errorf("process metrics: %w", err)
			} else {
				metrics.Process = processMetrics
			}
		}()
	}

	// Temperature metrics (optional)
	if c.temperatureEnabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if tempMetrics, err := c.collectTemperatureMetrics(ctx); err != nil {
				c.logger.Warn("Failed to collect temperature metrics", zap.Error(err))
			} else {
				metrics.Temperature = tempMetrics
			}
		}()
	}

	wg.Wait()
	close(errCh)

	// Check for errors
	var errors []error
	for err := range errCh {
		errors = append(errors, err)
		c.logger.Warn("Metric collection error", zap.Error(err))
	}

	// Return partial metrics even if some collectors failed
	if len(errors) > 0 && len(errors) == runtime.GOMAXPROCS(0) {
		return nil, fmt.Errorf("all metric collectors failed")
	}

	return metrics, nil
}

// collectCPUMetrics collects CPU-related metrics
func (c *SystemCollector) collectCPUMetrics(ctx context.Context) (types.CPUMetrics, error) {
	// Get CPU count
	logicalCount, err := cpu.CountsWithContext(ctx, true)
	if err != nil {
		return types.CPUMetrics{}, err
	}

	// Get overall CPU usage
	cpuPercents, err := cpu.PercentWithContext(ctx, time.Second, false)
	if err != nil {
		return types.CPUMetrics{}, err
	}

	// Get per-core usage
	perCorePercents, err := cpu.PercentWithContext(ctx, time.Second, true)
	if err != nil {
		return types.CPUMetrics{}, err
	}

	// Get detailed CPU times
	cpuTimes, err := cpu.TimesWithContext(ctx, false)
	if err != nil {
		return types.CPUMetrics{}, err
	}

	// Get load average
	loadAvg, err := load.AvgWithContext(ctx)
	if err != nil {
		c.logger.Warn("Failed to get load average", zap.Error(err))
		loadAvg = &load.AvgStat{} // Use empty struct as fallback
	}

	var cpuUsage float64
	if len(cpuPercents) > 0 {
		cpuUsage = cpuPercents[0]
	}

	var userPercent, systemPercent, idlePercent, iowaitPercent float64
	if len(cpuTimes) > 0 {
		total := cpuTimes[0].Total()
		if total > 0 {
			userPercent = cpuTimes[0].User / total * 100
			systemPercent = cpuTimes[0].System / total * 100
			idlePercent = cpuTimes[0].Idle / total * 100
			iowaitPercent = cpuTimes[0].Iowait / total * 100
		}
	}

	return types.CPUMetrics{
		UsagePercent:   cpuUsage,
		CoreCount:      logicalCount,
		PerCoreUsage:   perCorePercents,
		LoadAverage1m:  loadAvg.Load1,
		LoadAverage5m:  loadAvg.Load5,
		LoadAverage15m: loadAvg.Load15,
		UserPercent:    userPercent,
		SystemPercent:  systemPercent,
		IdlePercent:    idlePercent,
		IOWaitPercent:  iowaitPercent,
	}, nil
}

// collectMemoryMetrics collects memory-related metrics
func (c *SystemCollector) collectMemoryMetrics(ctx context.Context) (types.MemoryMetrics, error) {
	// Virtual memory
	vmStat, err := mem.VirtualMemoryWithContext(ctx)
	if err != nil {
		return types.MemoryMetrics{}, err
	}

	// Swap memory
	swapStat, err := mem.SwapMemoryWithContext(ctx)
	if err != nil {
		c.logger.Warn("Failed to get swap memory stats", zap.Error(err))
		swapStat = &mem.SwapMemoryStat{} // Use empty struct as fallback
	}

	return types.MemoryMetrics{
		TotalBytes:     vmStat.Total,
		AvailableBytes: vmStat.Available,
		UsedBytes:      vmStat.Used,
		FreeBytes:      vmStat.Free,
		UsagePercent:   vmStat.UsedPercent,
		BufferedBytes:  vmStat.Buffers,
		CachedBytes:    vmStat.Cached,
		SwapTotal:      swapStat.Total,
		SwapUsed:       swapStat.Used,
		SwapFree:       swapStat.Free,
		SwapPercent:    swapStat.UsedPercent,
	}, nil
}

// collectDiskMetrics collects disk-related metrics
func (c *SystemCollector) collectDiskMetrics(ctx context.Context) ([]types.DiskMetrics, error) {
	// Get disk usage for all partitions
	partitions, err := disk.PartitionsWithContext(ctx, false)
	if err != nil {
		return nil, err
	}

	// Get disk I/O stats
	ioStats, err := disk.IOCountersWithContext(ctx)
	if err != nil {
		c.logger.Warn("Failed to get disk I/O stats", zap.Error(err))
		ioStats = make(map[string]disk.IOCountersStat)
	}

	var diskMetrics []types.DiskMetrics

	for _, partition := range partitions {
		// Skip if mountpoints filter is set and this mountpoint is not in it
		if len(c.diskMountpoints) > 0 && !contains(c.diskMountpoints, partition.Mountpoint) {
			continue
		}

		usage, err := disk.UsageWithContext(ctx, partition.Mountpoint)
		if err != nil {
			c.logger.Warn("Failed to get disk usage",
				zap.String("mountpoint", partition.Mountpoint),
				zap.Error(err),
			)
			continue
		}

		diskMetric := types.DiskMetrics{
			Device:       partition.Device,
			Mountpoint:   partition.Mountpoint,
			Filesystem:   partition.Fstype,
			TotalBytes:   usage.Total,
			UsedBytes:    usage.Used,
			FreeBytes:    usage.Free,
			UsagePercent: usage.UsedPercent,
			InodesTotal:  usage.InodesTotal,
			InodesUsed:   usage.InodesUsed,
			InodesFree:   usage.InodesFree,
		}

		// Add I/O stats if available
		if ioStat, exists := ioStats[partition.Device]; exists {
			diskMetric.ReadCount = ioStat.ReadCount
			diskMetric.WriteCount = ioStat.WriteCount
			diskMetric.ReadBytes = ioStat.ReadBytes
			diskMetric.WriteBytes = ioStat.WriteBytes
			diskMetric.ReadTime = ioStat.ReadTime
			diskMetric.WriteTime = ioStat.WriteTime
			diskMetric.IOPercent = ioStat.IoTime / 1000.0 // Convert to percentage
		}

		diskMetrics = append(diskMetrics, diskMetric)
	}

	// Update last disk stats for rate calculations
	c.mu.Lock()
	c.lastDiskStats = ioStats
	c.mu.Unlock()

	return diskMetrics, nil
}

// collectNetworkMetrics collects network interface metrics
func (c *SystemCollector) collectNetworkMetrics(ctx context.Context) ([]types.NetworkMetrics, error) {
	// Get network I/O stats
	ioStats, err := net.IOCountersWithContext(ctx, true)
	if err != nil {
		return nil, err
	}

	// Get network interface info
	interfaces, err := net.InterfacesWithContext(ctx)
	if err != nil {
		c.logger.Warn("Failed to get network interfaces", zap.Error(err))
		interfaces = []net.InterfaceStat{}
	}

	// Create interface info map
	interfaceInfo := make(map[string]net.InterfaceStat)
	for _, iface := range interfaces {
		interfaceInfo[iface.Name] = iface
	}

	var networkMetrics []types.NetworkMetrics

	for _, ioStat := range ioStats {
		// Skip if interface filter is set and this interface is not in it
		if len(c.networkInterfaces) > 0 && !contains(c.networkInterfaces, ioStat.Name) {
			continue
		}

		netMetric := types.NetworkMetrics{
			Interface:     ioStat.Name,
			BytesSent:     ioStat.BytesSent,
			BytesReceived: ioStat.BytesRecv,
			PacketsSent:   ioStat.PacketsSent,
			PacketsRecv:   ioStat.PacketsRecv,
			ErrorsIn:      ioStat.Errin,
			ErrorsOut:     ioStat.Errout,
			DropsIn:       ioStat.Dropin,
			DropsOut:      ioStat.Dropout,
		}

		// Add interface info if available
		if info, exists := interfaceInfo[ioStat.Name]; exists {
			netMetric.MTU = info.MTU
			netMetric.IsUp = (info.Flags&1 != 0) // Check if IFF_UP flag is set

			// Try to get speed (this might not work on all systems)
			if len(info.Addrs) > 0 {
				// Speed information might not be available directly
				// This is a placeholder that could be enhanced with platform-specific code
				netMetric.Speed = 1000 // Default to 1Gbps
			}
		}

		networkMetrics = append(networkMetrics, netMetric)
	}

	// Update last network stats for rate calculations
	c.mu.Lock()
	c.lastNetworkStats = make(map[string]net.IOCountersStat)
	for _, ioStat := range ioStats {
		c.lastNetworkStats[ioStat.Name] = ioStat
	}
	c.mu.Unlock()

	return networkMetrics, nil
}

// collectLoadMetrics collects system load metrics
func (c *SystemCollector) collectLoadMetrics(ctx context.Context) (types.LoadMetrics, error) {
	loadAvg, err := load.AvgWithContext(ctx)
	if err != nil {
		return types.LoadMetrics{}, err
	}

	return types.LoadMetrics{
		Load1:  loadAvg.Load1,
		Load5:  loadAvg.Load5,
		Load15: loadAvg.Load15,
	}, nil
}

// collectProcessMetrics collects process-specific metrics
func (c *SystemCollector) collectProcessMetrics(ctx context.Context) ([]types.ProcessMetrics, error) {
	processes, err := process.ProcessesWithContext(ctx)
	if err != nil {
		return nil, err
	}

	var processMetrics []types.ProcessMetrics

	for _, proc := range processes {
		// Skip if we have filters and process doesn't match
		if len(c.processFilters) > 0 {
			name, err := proc.NameWithContext(ctx)
			if err != nil || !matchesFilter(name, c.processFilters) {
				continue
			}
		}

		procMetric, err := c.collectSingleProcessMetrics(ctx, proc)
		if err != nil {
			c.logger.Debug("Failed to collect process metrics",
				zap.Int32("pid", proc.Pid),
				zap.Error(err),
			)
			continue
		}

		processMetrics = append(processMetrics, procMetric)
	}

	return processMetrics, nil
}

// collectSingleProcessMetrics collects metrics for a single process
func (c *SystemCollector) collectSingleProcessMetrics(ctx context.Context, proc *process.Process) (types.ProcessMetrics, error) {
	name, err := proc.NameWithContext(ctx)
	if err != nil {
		name = "unknown"
	}

	status, err := proc.StatusWithContext(ctx)
	if err != nil {
		status = []string{"unknown"}
	}

	username, err := proc.UsernameWithContext(ctx)
	if err != nil {
		username = "unknown"
	}

	cpuPercent, err := proc.CPUPercentWithContext(ctx)
	if err != nil {
		cpuPercent = 0
	}

	memoryPercent, err := proc.MemoryPercentWithContext(ctx)
	if err != nil {
		memoryPercent = 0
	}

	memoryInfo, err := proc.MemoryInfoWithContext(ctx)
	if err != nil {
		memoryInfo = &process.MemoryInfoStat{}
	}

	numThreads, err := proc.NumThreadsWithContext(ctx)
	if err != nil {
		numThreads = 0
	}

	numFDs, err := proc.NumFDsWithContext(ctx)
	if err != nil {
		numFDs = 0
	}

	createTime, err := proc.CreateTimeWithContext(ctx)
	if err != nil {
		createTime = 0
	}

	openFiles, err := proc.OpenFilesWithContext(ctx)
	if err != nil {
		openFiles = []process.OpenFilesStat{}
	}

	connections, err := proc.ConnectionsWithContext(ctx)
	if err != nil {
		connections = []net.ConnectionStat{}
	}

	var statusStr string
	if len(status) > 0 {
		statusStr = status[0]
	}

	return types.ProcessMetrics{
		PID:           proc.Pid,
		Name:          name,
		Username:      username,
		Status:        statusStr,
		CPUPercent:    cpuPercent,
		MemoryPercent: float64(memoryPercent),
		MemoryRSS:     memoryInfo.RSS,
		MemoryVMS:     memoryInfo.VMS,
		NumThreads:    numThreads,
		NumFDs:        numFDs,
		CreateTime:    createTime,
		OpenFiles:     len(openFiles),
		Connections:   len(connections),
	}, nil
}

// collectTemperatureMetrics collects temperature sensor metrics
func (c *SystemCollector) collectTemperatureMetrics(ctx context.Context) ([]types.TemperatureMetrics, error) {
	temps, err := host.SensorsTemperaturesWithContext(ctx)
	if err != nil {
		return nil, err
	}

	var tempMetrics []types.TemperatureMetrics
	for _, temp := range temps {
		tempMetrics = append(tempMetrics, types.TemperatureMetrics{
			SensorKey:   temp.SensorKey,
			Temperature: temp.Temperature,
			High:        temp.High,
			Critical:    temp.Critical,
		})
	}

	return tempMetrics, nil
}

// Helper functions

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func matchesFilter(name string, filters []string) bool {
	if len(filters) == 0 {
		return true
	}
	
	for _, filter := range filters {
		if name == filter {
			return true
		}
	}
	return false
}