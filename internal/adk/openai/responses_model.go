package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var _ model.LLM = &ResponsesModel{}

// HTTPDoer HTTP 客户端接口（与 go-openai 兼容）
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// ResponsesModel 实现 model.LLM 接口，使用 OpenAI Responses API
type ResponsesModel struct {
	httpClient HTTPDoer
	baseURL    string
	apiKey     string
	modelName  string
}

// NewResponsesModel 创建 Responses API 模型
// apiKey 从工厂单独传入，因 go-openai ClientConfig.authToken 不可导出
func NewResponsesModel(modelName, apiKey, baseURL string, httpClient HTTPDoer) *ResponsesModel {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ResponsesModel{
		httpClient: httpClient,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		modelName:  modelName,
	}
}

// Name 返回模型名称
func (r *ResponsesModel) Name() string {
	return r.modelName
}

// GenerateContent 实现 model.LLM 接口
func (r *ResponsesModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return r.generateStream(ctx, req)
	}
	return r.generate(ctx, req)
}

// responsesEndpoint 返回 Responses API 端点 URL
// baseURL 已由工厂层 normalizeOpenAIBaseURL 规范化，保证以 /v1 结尾
func (r *ResponsesModel) responsesEndpoint() string {
	return r.baseURL + "/responses"
}

// doRequest 发送 HTTP 请求到 Responses API
func (r *ResponsesModel) doRequest(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.responsesEndpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)
	if stream {
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("Cache-Control", "no-cache")
		req.Header.Set("Connection", "keep-alive")
	}
	return r.httpClient.Do(req)
}

// generate 非流式生成
func (r *ResponsesModel) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		apiReq, err := toResponsesRequest(req, r.modelName)
		if err != nil {
			yield(nil, err)
			return
		}
		apiReq.Stream = false

		body, err := json.Marshal(apiReq)
		if err != nil {
			yield(nil, fmt.Errorf("序列化请求失败: %w", err))
			return
		}

		resp, err := r.doRequest(ctx, body, false)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 400 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("Responses API 错误 (HTTP %d): %s", resp.StatusCode, string(respBody)))
			return
		}

		var apiResp CreateResponseResponse
		if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
			yield(nil, fmt.Errorf("解析响应失败: %w", err))
			return
		}

		llmResp, err := convertResponsesResponse(&apiResp)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(llmResp, nil)
	}
}

// generateStream 流式生成
func (r *ResponsesModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		apiReq, err := toResponsesRequest(req, r.modelName)
		if err != nil {
			yield(nil, err)
			return
		}
		apiReq.Stream = true

		body, err := json.Marshal(apiReq)
		if err != nil {
			yield(nil, fmt.Errorf("序列化请求失败: %w", err))
			return
		}

		resp, err := r.doRequest(ctx, body, true)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 400 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("Responses API 流式错误 (HTTP %d): %s", resp.StatusCode, string(respBody)))
			return
		}

		r.processResponsesStream(resp.Body, yield)
	}
}

// processResponsesStream 处理 Responses API 的 SSE 流
func (r *ResponsesModel) processResponsesStream(body io.Reader, yield func(*model.LLMResponse, error) bool) {
	scanner := bufio.NewScanner(body)
	// 聚合状态
	aggregatedContent := &genai.Content{Role: "model", Parts: []*genai.Part{}}
	var textContent string
	toolCallsMap := make(map[string]*responsesToolCallBuilder)
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	var currentEventType string

	for scanner.Scan() {
		line := scanner.Text()

		// SSE 格式解析: "event: <type>" 和 "data: <json>"
		if eventType, ok := strings.CutPrefix(line, "event: "); ok {
			currentEventType = eventType
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "" {
			continue
		}

		switch currentEventType {
		case "response.output_text.delta":
			r.handleTextDelta(data, &textContent, yield)

		case "response.function_call_arguments.delta":
			r.handleFuncArgsDelta(data, toolCallsMap)

		case "response.output_item.added":
			r.handleOutputItemAdded(data, toolCallsMap)

		case "response.output_item.done":
			r.handleOutputItemDone(data, toolCallsMap)

		case "response.completed":
			r.handleCompleted(data, &usageMetadata)
		}

		currentEventType = ""
	}

	// 组装最终聚合响应
	if textContent != "" {
		aggregatedContent.Parts = append(aggregatedContent.Parts, &genai.Part{Text: textContent})
	}
	for _, builder := range toolCallsMap {
		aggregatedContent.Parts = append(aggregatedContent.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   builder.callID,
				Name: builder.name,
				Args: parseJSONArgs(builder.args),
			},
		})
	}

	finalResp := &model.LLMResponse{
		Content:       aggregatedContent,
		UsageMetadata: usageMetadata,
		FinishReason:  genai.FinishReasonStop,
		Partial:       false,
		TurnComplete:  true,
	}
	yield(finalResp, nil)
}

// responsesToolCallBuilder 用于聚合流式工具调用
type responsesToolCallBuilder struct {
	itemID string
	callID string
	name   string
	args   string
}

// handleTextDelta 处理文本增量事件
func (r *ResponsesModel) handleTextDelta(data string, textContent *string, yield func(*model.LLMResponse, error) bool) {
	var delta ResponsesTextDelta
	if json.Unmarshal([]byte(data), &delta) != nil {
		return
	}
	*textContent += delta.Delta
	part := &genai.Part{Text: delta.Delta}
	llmResp := &model.LLMResponse{
		Content:      &genai.Content{Role: "model", Parts: []*genai.Part{part}},
		Partial:      true,
		TurnComplete: false,
	}
	yield(llmResp, nil)
}

// handleFuncArgsDelta 处理函数调用参数增量事件
func (r *ResponsesModel) handleFuncArgsDelta(data string, toolCallsMap map[string]*responsesToolCallBuilder) {
	var delta ResponsesFuncCallArgsDelta
	if json.Unmarshal([]byte(data), &delta) != nil {
		return
	}
	if builder, exists := toolCallsMap[delta.ItemID]; exists {
		builder.args += delta.Delta
	}
}

// handleOutputItemAdded 处理输出项添加事件
func (r *ResponsesModel) handleOutputItemAdded(data string, toolCallsMap map[string]*responsesToolCallBuilder) {
	var added ResponsesOutputItemAdded
	if json.Unmarshal([]byte(data), &added) != nil {
		return
	}
	if added.Item.Type == "function_call" {
		toolCallsMap[added.Item.ID] = &responsesToolCallBuilder{
			itemID: added.Item.ID,
			callID: added.Item.CallID,
			name:   added.Item.Name,
		}
	}
}

// handleOutputItemDone 处理输出项完成事件
func (r *ResponsesModel) handleOutputItemDone(data string, toolCallsMap map[string]*responsesToolCallBuilder) {
	var done ResponsesOutputItemDone
	if json.Unmarshal([]byte(data), &done) != nil {
		return
	}
	if done.Item.Type == "function_call" {
		if builder, exists := toolCallsMap[done.Item.ID]; exists {
			builder.callID = done.Item.CallID
			builder.name = done.Item.Name
			if done.Item.Arguments != "" {
				builder.args = done.Item.Arguments
			}
		} else {
			toolCallsMap[done.Item.ID] = &responsesToolCallBuilder{
				itemID: done.Item.ID,
				callID: done.Item.CallID,
				name:   done.Item.Name,
				args:   done.Item.Arguments,
			}
		}
	}
}

// handleCompleted 处理响应完成事件
func (r *ResponsesModel) handleCompleted(data string, usageMetadata **genai.GenerateContentResponseUsageMetadata) {
	var completed ResponsesCompleted
	if json.Unmarshal([]byte(data), &completed) != nil {
		return
	}
	if completed.Response.Usage != nil {
		*usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(completed.Response.Usage.InputTokens),
			CandidatesTokenCount: int32(completed.Response.Usage.OutputTokens),
			TotalTokenCount:      int32(completed.Response.Usage.TotalTokens),
		}
	}
}
