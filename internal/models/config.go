package models

// AIProvider AI服务提供商类型
type AIProvider string

const (
	AIProviderOpenAI   AIProvider = "openai"
	AIProviderGemini   AIProvider = "gemini"
	AIProviderVertexAI AIProvider = "vertexai"
)

// AIConfig AI服务配置
type AIConfig struct {
	ID               string     `json:"id"`
	Name             string     `json:"name"`
	Provider         AIProvider `json:"provider"`
	BaseURL          string     `json:"baseUrl"`
	APIKey           string     `json:"apiKey"`
	ModelName        string     `json:"modelName"`
	MaxTokens        int        `json:"maxTokens"`
	Temperature      float64    `json:"temperature"`
	Timeout          int        `json:"timeout"`
	HttpProxy        string     `json:"httpProxy"`
	HttpProxyEnabled bool       `json:"httpProxyEnabled"`
	IsDefault        bool       `json:"isDefault"`
	// OpenAI Responses API 开关
	UseResponses bool `json:"useResponses"` // 使用 Responses API 格式（/v1/responses）
	// Vertex AI 专用字段
	Project         string `json:"project"`         // GCP 项目 ID
	Location        string `json:"location"`        // GCP 区域，如 us-central1
	CredentialsJSON string `json:"credentialsJson"` // 服务账号 JSON 证书内容
}

// MCPTransportType MCP传输类型
type MCPTransportType string

const (
	MCPTransportHTTP    MCPTransportType = "http"    // StreamableHTTP 传输
	MCPTransportSSE     MCPTransportType = "sse"     // SSE 传输（已废弃）
	MCPTransportCommand MCPTransportType = "command" // 命令行传输
)

// MCPServerConfig MCP服务器配置
type MCPServerConfig struct {
	ID            string           `json:"id"`
	Name          string           `json:"name"`
	TransportType MCPTransportType `json:"transportType"`
	Endpoint      string           `json:"endpoint"`      // HTTP/SSE 端点 URL
	Command       string           `json:"command"`       // 命令行传输的命令
	Args          []string         `json:"args"`          // 命令行参数
	ToolFilter    []string         `json:"toolFilter"`    // 工具过滤列表（空则全部）
	Enabled       bool             `json:"enabled"`       // 是否启用
}

// AppConfig 应用配置
type AppConfig struct {
	RefreshInterval int64             `json:"refreshInterval"`
	Theme           string            `json:"theme"` // 主题色: military, ocean, purple, orange, dark
	AIConfigs       []AIConfig        `json:"aiConfigs"`
	DefaultAIID     string            `json:"defaultAiId"`
	MCPServers      []MCPServerConfig `json:"mcpServers"` // MCP服务器配置列表
	Memory          MemoryConfig      `json:"memory"`     // 记忆管理配置
}

// MemoryConfig 记忆管理配置
type MemoryConfig struct {
	Enabled           bool   `json:"enabled"`           // 是否启用记忆管理
	AIConfigID        string `json:"aiConfigId"`        // 使用的 LLM 配置 ID（空则使用默认）
	MaxRecentRounds   int    `json:"maxRecentRounds"`   // 保留最近几轮讨论
	MaxKeyFacts       int    `json:"maxKeyFacts"`       // 最大关键事实数
	MaxSummaryLength  int    `json:"maxSummaryLength"`  // 摘要最大字数
	CompressThreshold int    `json:"compressThreshold"` // 触发压缩的轮次数
}
