package adk

import (
	"context"
	"fmt"
	"strings"

	"cloud.google.com/go/auth/credentials"
	"github.com/run-bigpig/jcp/internal/adk/openai"
	"github.com/run-bigpig/jcp/internal/models"

	go_openai "github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/genai"
)

// ModelFactory 模型工厂，根据配置创建对应的 adk model
type ModelFactory struct{}

// NewModelFactory 创建模型工厂
func NewModelFactory() *ModelFactory {
	return &ModelFactory{}
}

// CreateModel 根据 AI 配置创建对应的模型
func (f *ModelFactory) CreateModel(ctx context.Context, config *models.AIConfig) (model.LLM, error) {
	switch config.Provider {
	case models.AIProviderGemini:
		return f.createGeminiModel(ctx, config)
	case models.AIProviderVertexAI:
		return f.createVertexAIModel(ctx, config)
	case models.AIProviderOpenAI:
		if config.UseResponses {
			return f.createOpenAIResponsesModel(config)
		}
		return f.createOpenAIModel(config)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", config.Provider)
	}
}

// createGeminiModel 创建 Gemini 模型
func (f *ModelFactory) createGeminiModel(ctx context.Context, config *models.AIConfig) (model.LLM, error) {
	clientConfig := &genai.ClientConfig{
		APIKey:  config.APIKey,
		Backend: genai.BackendGeminiAPI,
	}

	return gemini.NewModel(ctx, config.ModelName, clientConfig)
}

// createVertexAIModel 创建 Vertex AI 模型
func (f *ModelFactory) createVertexAIModel(ctx context.Context, config *models.AIConfig) (model.LLM, error) {
	clientConfig := &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  config.Project,
		Location: config.Location,
	}

	// 如果提供了证书 JSON，则使用证书认证
	if config.CredentialsJSON != "" {
		creds, err := credentials.DetectDefault(&credentials.DetectOptions{
			Scopes:          []string{"https://www.googleapis.com/auth/cloud-platform"},
			CredentialsJSON: []byte(config.CredentialsJSON),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create credentials: %w", err)
		}
		clientConfig.Credentials = creds
	}

	return gemini.NewModel(ctx, config.ModelName, clientConfig)
}

// normalizeOpenAIBaseURL 规范化 OpenAI BaseURL
// 确保 URL 以 /v1 结尾，兼容用户填写带或不带 /v1 的地址
func normalizeOpenAIBaseURL(baseURL string) string {
	if baseURL == "" {
		return "https://api.openai.com/v1"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	return baseURL
}

// createOpenAIModel 创建 OpenAI 兼容模型
func (f *ModelFactory) createOpenAIModel(config *models.AIConfig) (model.LLM, error) {
	openaiCfg := go_openai.DefaultConfig(config.APIKey)
	openaiCfg.BaseURL = normalizeOpenAIBaseURL(config.BaseURL)

	return openai.NewOpenAIModel(config.ModelName, openaiCfg), nil
}

// createOpenAIResponsesModel 创建使用 Responses API 的 OpenAI 模型
func (f *ModelFactory) createOpenAIResponsesModel(config *models.AIConfig) (model.LLM, error) {
	baseURL := normalizeOpenAIBaseURL(config.BaseURL)

	// 复用 go-openai 的 HTTPClient 以保持代理/超时一致
	openaiCfg := go_openai.DefaultConfig(config.APIKey)
	return openai.NewResponsesModel(config.ModelName, config.APIKey, baseURL, openaiCfg.HTTPClient), nil
}
