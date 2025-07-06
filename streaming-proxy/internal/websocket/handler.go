package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"

	"smartrag-streaming-proxy/internal/config"
	"smartrag-streaming-proxy/internal/proxy"
	"smartrag-streaming-proxy/pkg/types"
)

// Manager manages WebSocket connections and message routing
type Manager struct {
	config      *config.StreamingConfig
	logger      *zap.Logger
	proxy       *proxy.StreamingProxy
	upgrader    websocket.Upgrader
	connections map[string]*Connection
	mu          sync.RWMutex
	hub         chan *Message
	register    chan *Connection
	unregister  chan *Connection
	shutdown    chan struct{}
	stats       *Stats
}

// Connection represents a WebSocket client connection
type Connection struct {
	ID         string
	conn       *websocket.Conn
	manager    *Manager
	send       chan []byte
	sessionID  string
	clientIP   string
	userAgent  string
	connectedAt time.Time
	lastPing   time.Time
	ctx        context.Context
	cancel     context.CancelFunc
}

// Message represents a message to be sent to connections
type Message struct {
	ConnectionIDs []string // empty means broadcast to all
	Data         []byte
	MessageType  int // websocket.TextMessage or websocket.BinaryMessage
}

// Stats holds WebSocket statistics
type Stats struct {
	mu               sync.RWMutex
	ActiveConnections int
	TotalConnections int64
	MessagesProcessed int64
	BytesTransferred int64
	ErrorCount       int64
}

// NewManager creates a new WebSocket manager
func NewManager(cfg *config.StreamingConfig, logger *zap.Logger, proxy *proxy.StreamingProxy) *Manager {
	upgrader := websocket.Upgrader{
		ReadBufferSize:   cfg.WebSocket.ReadBufferSize,
		WriteBufferSize:  cfg.WebSocket.WriteBufferSize,
		HandshakeTimeout: cfg.WebSocket.HandshakeTimeout,
		EnableCompression: cfg.WebSocket.EnableCompression,
		Subprotocols:     cfg.WebSocket.Subprotocols,
		CheckOrigin: func(r *http.Request) bool {
			if !cfg.WebSocket.CheckOrigin {
				return true
			}
			// Implement origin checking logic here
			return true
		},
	}

	return &Manager{
		config:      cfg,
		logger:      logger,
		proxy:       proxy,
		upgrader:    upgrader,
		connections: make(map[string]*Connection),
		hub:         make(chan *Message, 256),
		register:    make(chan *Connection),
		unregister:  make(chan *Connection),
		shutdown:    make(chan struct{}),
		stats:       &Stats{},
	}
}

// Start starts the WebSocket manager
func (m *Manager) Start() {
	go m.run()
	m.logger.Info("WebSocket manager started")
}

// Stop stops the WebSocket manager
func (m *Manager) Stop() {
	close(m.shutdown)
	m.logger.Info("WebSocket manager stopped")
}

// HandleConnection handles WebSocket connection upgrade
func (m *Manager) HandleConnection(c *gin.Context) {
	// Check connection limits
	if m.getActiveConnections() >= m.config.MaxClients {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Maximum connections exceeded",
		})
		return
	}

	// Upgrade connection
	conn, err := m.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		m.logger.Error("Failed to upgrade connection", zap.Error(err))
		return
	}

	// Create connection
	ctx, cancel := context.WithCancel(context.Background())
	connection := &Connection{
		ID:          generateConnectionID(),
		conn:        conn,
		manager:     m,
		send:        make(chan []byte, 256),
		sessionID:   c.Query("session_id"),
		clientIP:    c.ClientIP(),
		userAgent:   c.Request.UserAgent(),
		connectedAt: time.Now(),
		lastPing:    time.Now(),
		ctx:         ctx,
		cancel:      cancel,
	}

	// Register connection
	m.register <- connection

	// Start goroutines for handling the connection
	go connection.writePump()
	go connection.readPump()

	m.logger.Info("WebSocket connection established",
		zap.String("connection_id", connection.ID),
		zap.String("client_ip", connection.clientIP),
		zap.String("session_id", connection.sessionID),
	)
}

// run runs the main message hub
func (m *Manager) run() {
	ticker := time.NewTicker(m.config.PingInterval)
	defer ticker.Stop()

	for {
		select {
		case connection := <-m.register:
			m.addConnection(connection)

		case connection := <-m.unregister:
			m.removeConnection(connection)

		case message := <-m.hub:
			m.broadcastMessage(message)

		case <-ticker.C:
			m.pingConnections()

		case <-m.shutdown:
			m.closeAllConnections()
			return
		}
	}
}

// addConnection adds a connection to the manager
func (m *Manager) addConnection(conn *Connection) {
	m.mu.Lock()
	m.connections[conn.ID] = conn
	m.stats.mu.Lock()
	m.stats.ActiveConnections++
	m.stats.TotalConnections++
	m.stats.mu.Unlock()
	m.mu.Unlock()

	m.logger.Debug("Connection added",
		zap.String("connection_id", conn.ID),
		zap.Int("active_connections", len(m.connections)),
	)
}

// removeConnection removes a connection from the manager
func (m *Manager) removeConnection(conn *Connection) {
	m.mu.Lock()
	if _, exists := m.connections[conn.ID]; exists {
		delete(m.connections, conn.ID)
		close(conn.send)
		conn.cancel()
		m.stats.mu.Lock()
		m.stats.ActiveConnections--
		m.stats.mu.Unlock()
	}
	m.mu.Unlock()

	m.logger.Debug("Connection removed",
		zap.String("connection_id", conn.ID),
		zap.Int("active_connections", len(m.connections)),
	)
}

// broadcastMessage broadcasts a message to specified connections or all
func (m *Manager) broadcastMessage(message *Message) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// If no specific connection IDs, broadcast to all
	if len(message.ConnectionIDs) == 0 {
		for _, conn := range m.connections {
			select {
			case conn.send <- message.Data:
				m.updateBytesSent(len(message.Data))
			default:
				// Connection's send channel is full, close it
				m.unregister <- conn
			}
		}
		return
	}

	// Send to specific connections
	for _, connID := range message.ConnectionIDs {
		if conn, exists := m.connections[connID]; exists {
			select {
			case conn.send <- message.Data:
				m.updateBytesSent(len(message.Data))
			default:
				m.unregister <- conn
			}
		}
	}
}

// pingConnections sends ping to all connections
func (m *Manager) pingConnections() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	now := time.Now()
	for _, conn := range m.connections {
		if now.Sub(conn.lastPing) > m.config.ClientTimeout {
			m.logger.Warn("Connection timeout, closing",
				zap.String("connection_id", conn.ID),
				zap.Duration("last_ping", now.Sub(conn.lastPing)),
			)
			m.unregister <- conn
			continue
		}

		// Send ping
		select {
		case conn.send <- []byte(`{"type":"ping","timestamp":"` + now.Format(time.RFC3339) + `"}`):
		default:
			m.unregister <- conn
		}
	}
}

// closeAllConnections closes all active connections
func (m *Manager) closeAllConnections() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, conn := range m.connections {
		conn.cancel()
		conn.conn.Close()
	}
	m.connections = make(map[string]*Connection)
	m.stats.mu.Lock()
	m.stats.ActiveConnections = 0
	m.stats.mu.Unlock()
}

// readPump pumps messages from the websocket connection
func (c *Connection) readPump() {
	defer func() {
		c.manager.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(c.manager.config.MaxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(c.manager.config.ClientTimeout))
	c.conn.SetPongHandler(func(string) error {
		c.lastPing = time.Now()
		c.conn.SetReadDeadline(time.Now().Add(c.manager.config.ClientTimeout))
		return nil
	})

	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		messageType, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.manager.logger.Error("WebSocket error", zap.Error(err))
			}
			break
		}

		if messageType == websocket.TextMessage {
			if err := c.handleMessage(data); err != nil {
				c.manager.logger.Error("Failed to handle message",
					zap.String("connection_id", c.ID),
					zap.Error(err),
				)
			}
		}
	}
}

// writePump pumps messages from the hub to the websocket connection
func (c *Connection) writePump() {
	ticker := time.NewTicker(c.manager.config.PingInterval)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case data, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
				c.manager.logger.Error("Failed to write message",
					zap.String("connection_id", c.ID),
					zap.Error(err),
				)
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}

		case <-c.ctx.Done():
			return
		}
	}
}

// handleMessage handles incoming WebSocket messages
func (c *Connection) handleMessage(data []byte) error {
	var clientMsg types.ClientMessage
	if err := json.Unmarshal(data, &clientMsg); err != nil {
		return fmt.Errorf("invalid message format: %w", err)
	}

	switch clientMsg.Type {
	case "query":
		return c.handleStreamingQuery(clientMsg.Payload)
	case "ping":
		c.lastPing = time.Now()
		pongMsg := types.ServerMessage{
			Type:      "pong",
			Timestamp: time.Now(),
		}
		if pongData, err := json.Marshal(pongMsg); err == nil {
			select {
			case c.send <- pongData:
			default:
			}
		}
	case "close":
		c.cancel()
	}

	return nil
}

// handleStreamingQuery handles streaming query requests
func (c *Connection) handleStreamingQuery(payload json.RawMessage) error {
	var request types.StreamRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		return fmt.Errorf("invalid query request: %w", err)
	}

	// Set streaming flag
	request.Stream = true
	if request.SessionID == "" {
		request.SessionID = c.sessionID
	}

	// Start streaming from backend
	stream, err := c.manager.proxy.StartStream(c.ctx, &request)
	if err != nil {
		c.sendError("stream_start_failed", err.Error())
		return err
	}

	// Handle streaming response
	go c.handleStreamingResponse(stream)

	return nil
}

// handleStreamingResponse handles streaming responses from backend
func (c *Connection) handleStreamingResponse(stream <-chan types.StreamResponse) {
	for {
		select {
		case response, ok := <-stream:
			if !ok {
				// Stream closed
				c.sendMessage("done", nil)
				return
			}

			// Forward response to client
			serverMsg := types.ServerMessage{
				Type:      response.Type,
				Timestamp: time.Now(),
			}

			if data, err := json.Marshal(response); err == nil {
				serverMsg.Payload = data
			}

			if msgData, err := json.Marshal(serverMsg); err == nil {
				select {
				case c.send <- msgData:
					c.manager.updateMessagesProcessed()
				default:
					// Send channel full, connection will be closed
					return
				}
			}

		case <-c.ctx.Done():
			return
		}
	}
}

// sendMessage sends a message to the client
func (c *Connection) sendMessage(msgType string, payload interface{}) {
	serverMsg := types.ServerMessage{
		Type:      msgType,
		Timestamp: time.Now(),
	}

	if payload != nil {
		if data, err := json.Marshal(payload); err == nil {
			serverMsg.Payload = data
		}
	}

	if msgData, err := json.Marshal(serverMsg); err == nil {
		select {
		case c.send <- msgData:
		default:
		}
	}
}

// sendError sends an error message to the client
func (c *Connection) sendError(code, message string) {
	errorDetails := types.ErrorDetails{
		Code:    code,
		Message: message,
	}
	c.sendMessage("error", errorDetails)
}

// Helper functions

func generateConnectionID() string {
	return fmt.Sprintf("ws_%d_%d", time.Now().UnixNano(), time.Now().Nanosecond())
}

func (m *Manager) getActiveConnections() int {
	m.stats.mu.RLock()
	defer m.stats.mu.RUnlock()
	return m.stats.ActiveConnections
}

func (m *Manager) updateBytesSent(bytes int) {
	m.stats.mu.Lock()
	m.stats.BytesTransferred += int64(bytes)
	m.stats.mu.Unlock()
}

func (m *Manager) updateMessagesProcessed() {
	m.stats.mu.Lock()
	m.stats.MessagesProcessed++
	m.stats.mu.Unlock()
}

// GetStats returns current WebSocket statistics
func (m *Manager) GetStats() *Stats {
	m.stats.mu.RLock()
	defer m.stats.mu.RUnlock()
	
	return &Stats{
		ActiveConnections: m.stats.ActiveConnections,
		TotalConnections:  m.stats.TotalConnections,
		MessagesProcessed: m.stats.MessagesProcessed,
		BytesTransferred:  m.stats.BytesTransferred,
		ErrorCount:        m.stats.ErrorCount,
	}
}