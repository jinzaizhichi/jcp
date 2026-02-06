package services

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/run-bigpig/jcp/internal/models"

	"github.com/google/uuid"
)

// SessionService Session服务
type SessionService struct {
	sessionsDir string
	sessions    map[string]*models.StockSession
	mu          sync.RWMutex
}

// NewSessionService 创建Session服务
func NewSessionService(dataDir string) *SessionService {
	ss := &SessionService{
		sessionsDir: filepath.Join(dataDir, "sessions"),
		sessions:    make(map[string]*models.StockSession),
	}
	ss.ensureDir()
	return ss
}

// ensureDir 确保目录存在
func (ss *SessionService) ensureDir() {
	if err := os.MkdirAll(ss.sessionsDir, 0755); err != nil {
		fmt.Printf("创建sessions目录失败: %v\n", err)
	}
}

// getSessionPath 获取Session文件路径
func (ss *SessionService) getSessionPath(stockCode string) string {
	return filepath.Join(ss.sessionsDir, stockCode+".json")
}

// GetOrCreateSession 获取或创建Session
func (ss *SessionService) GetOrCreateSession(stockCode, stockName string) (*models.StockSession, error) {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	// 先从内存缓存获取
	if session, ok := ss.sessions[stockCode]; ok {
		return session, nil
	}

	// 尝试从文件加载
	session, err := ss.loadSession(stockCode)
	if err == nil {
		ss.sessions[stockCode] = session
		return session, nil
	}

	// 创建新Session
	now := time.Now().UnixMilli()
	session = &models.StockSession{
		ID:        uuid.New().String(),
		StockCode: stockCode,
		StockName: stockName,
		Messages:  []models.ChatMessage{},
		CreatedAt: now,
		UpdatedAt: now,
	}

	ss.sessions[stockCode] = session
	return session, ss.saveSession(session)
}

// loadSession 从文件加载Session
func (ss *SessionService) loadSession(stockCode string) (*models.StockSession, error) {
	path := ss.getSessionPath(stockCode)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var session models.StockSession
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, err
	}
	return &session, nil
}

// saveSession 保存Session到文件
func (ss *SessionService) saveSession(session *models.StockSession) error {
	path := ss.getSessionPath(session.StockCode)
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// GetSession 获取Session
func (ss *SessionService) GetSession(stockCode string) *models.StockSession {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	// 先从内存缓存获取
	if session, ok := ss.sessions[stockCode]; ok {
		return session
	}

	// 内存没有则尝试从文件加载
	session, err := ss.loadSession(stockCode)
	if err != nil {
		return nil
	}

	ss.sessions[stockCode] = session
	return session
}

// AddMessage 添加消息到Session
func (ss *SessionService) AddMessage(stockCode string, msg models.ChatMessage) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	session, ok := ss.sessions[stockCode]
	if !ok {
		// 尝试从文件加载
		var err error
		session, err = ss.loadSession(stockCode)
		if err != nil {
			return fmt.Errorf("session not found: %s", stockCode)
		}
		ss.sessions[stockCode] = session
	}

	msg.ID = uuid.New().String()
	msg.Timestamp = time.Now().UnixMilli()
	session.Messages = append(session.Messages, msg)
	session.UpdatedAt = time.Now().UnixMilli()
	return ss.saveSession(session)
}

// AddMessages 批量添加消息到Session
func (ss *SessionService) AddMessages(stockCode string, msgs []models.ChatMessage) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	session, ok := ss.sessions[stockCode]
	if !ok {
		// 尝试从文件加载
		var err error
		session, err = ss.loadSession(stockCode)
		if err != nil {
			return fmt.Errorf("session not found: %s", stockCode)
		}
		ss.sessions[stockCode] = session
	}

	now := time.Now().UnixMilli()
	for i := range msgs {
		msgs[i].ID = uuid.New().String()
		msgs[i].Timestamp = now
	}
	session.Messages = append(session.Messages, msgs...)
	session.UpdatedAt = now
	return ss.saveSession(session)
}

// GetMessages 获取Session消息
func (ss *SessionService) GetMessages(stockCode string) []models.ChatMessage {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	// 先从内存缓存获取
	if session, ok := ss.sessions[stockCode]; ok {
		return session.Messages
	}

	// 内存没有则尝试从文件加载
	session, err := ss.loadSession(stockCode)
	if err != nil {
		return []models.ChatMessage{}
	}

	// 加载成功后缓存到内存
	ss.sessions[stockCode] = session
	return session.Messages
}

// ClearMessages 清空Session消息
func (ss *SessionService) ClearMessages(stockCode string) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	session, ok := ss.sessions[stockCode]
	if !ok {
		// 尝试从文件加载
		var err error
		session, err = ss.loadSession(stockCode)
		if err != nil {
			return fmt.Errorf("session not found: %s", stockCode)
		}
		ss.sessions[stockCode] = session
	}

	session.Messages = []models.ChatMessage{}
	session.UpdatedAt = time.Now().UnixMilli()
	return ss.saveSession(session)
}

// UpdatePosition 更新持仓信息
func (ss *SessionService) UpdatePosition(stockCode string, shares int64, costPrice float64) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	session, ok := ss.sessions[stockCode]
	if !ok {
		// 尝试从文件加载
		var err error
		session, err = ss.loadSession(stockCode)
		if err != nil {
			return fmt.Errorf("session not found: %s", stockCode)
		}
		ss.sessions[stockCode] = session
	}

	session.Position = &models.StockPosition{
		Shares:    shares,
		CostPrice: costPrice,
	}
	session.UpdatedAt = time.Now().UnixMilli()
	return ss.saveSession(session)
}

// GetPosition 获取持仓信息
func (ss *SessionService) GetPosition(stockCode string) *models.StockPosition {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	session, ok := ss.sessions[stockCode]
	if !ok {
		// 尝试从文件加载
		session, err := ss.loadSession(stockCode)
		if err != nil {
			return nil
		}
		ss.sessions[stockCode] = session
		return session.Position
	}
	return session.Position
}
