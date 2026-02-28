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

	"github.com/run-bigpig/jcp/internal/logger"
)

var respLog = logger.New("openai:responses")

// sseMaxBufferSize SSE 扫描器最大缓冲区（1MB），防止超长工具参数被截断
const sseMaxBufferSize = 1024 * 1024

var _ model.LLM = &ResponsesModel{}

// HTTPDoer HTTP 客户端接口
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// ResponsesModel 实现 model.LLM 接口，使用 OpenAI Responses API
type ResponsesModel struct {
	httpClient   HTTPDoer
	baseURL      string
	apiKey       string
	modelName    string
	NoSystemRole bool // 不支持 system role 时需要降级处理
}

// NewResponsesModel 创建 Responses API 模型
func NewResponsesModel(modelName, apiKey, baseURL string, httpClient HTTPDoer, noSystemRole bool) *ResponsesModel {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ResponsesModel{
		httpClient:   httpClient,
		baseURL:      strings.TrimRight(baseURL, "/"),
		apiKey:       apiKey,
		modelName:    modelName,
		NoSystemRole: noSystemRole,
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
func (r *ResponsesModel) responsesEndpoint() string {
	return r.baseURL + "/responses"
}

// doRequest 发送 HTTP 请求
func (r *ResponsesModel) doRequest(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.responsesEndpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) CherryStudio/1.2.4 Chrome/126.0.6478.234 Electron/31.7.6 Safari/537.36")
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
		apiReq, err := toResponsesRequest(req, r.modelName, r.NoSystemRole)
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
		apiReq, err := toResponsesRequest(req, r.modelName, r.NoSystemRole)
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
	scanner.Buffer(make([]byte, 0, 64*1024), sseMaxBufferSize)

	aggregatedContent := &genai.Content{Role: "model", Parts: []*genai.Part{}}
	var textContent string
	var thoughtContent string
	toolCallsMap := make(map[string]*responsesToolCallBuilder)
	var toolCallOrder []string
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	var currentEventType string
	thinkParser := newThinkTagStreamParser()

	for scanner.Scan() {
		line := scanner.Text()

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
			if !r.handleTextDelta(data, thinkParser, &textContent, &thoughtContent, yield) {
				return
			}
		case "response.function_call_arguments.delta":
			r.handleFuncArgsDelta(data, toolCallsMap)
		case "response.output_item.added":
			r.handleOutputItemAdded(data, toolCallsMap, &toolCallOrder)
		case "response.output_item.done":
			r.handleOutputItemDone(data, toolCallsMap, &toolCallOrder)
		case "response.completed":
			r.handleCompleted(data, &usageMetadata)
		}

		currentEventType = ""
	}

	if err := scanner.Err(); err != nil {
		respLog.Warn("SSE 流读取错误: %v", err)
		yield(nil, fmt.Errorf("SSE 流读取错误: %w", err))
		return
	}

	// 刷新剩余分片（处理标签跨 chunk）
	if !r.emitTextSegments(thinkParser.Flush(), &textContent, &thoughtContent, yield) {
		return
	}

	// 组装最终文本，并解析第三方工具调用标记
	if textContent != "" {
		vendorCalls, cleanedText := parseVendorToolCalls(textContent)
		if cleanedText != "" {
			aggregatedContent.Parts = append(aggregatedContent.Parts, &genai.Part{Text: cleanedText})
		}
		for i, vc := range vendorCalls {
			aggregatedContent.Parts = append(aggregatedContent.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   fmt.Sprintf("vendor_call_%d", i),
					Name: vc.Name,
					Args: vc.Args,
				},
			})
		}
	}

	// 按插入顺序输出标准工具调用
	for _, id := range toolCallOrder {
		builder := toolCallsMap[id]
		if builder == nil {
			continue
		}
		aggregatedContent.Parts = append(aggregatedContent.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   builder.callID,
				Name: builder.name,
				Args: parseJSONArgs(builder.args),
			},
		})
	}

	if thoughtContent != "" {
		aggregatedContent.Parts = append([]*genai.Part{{Text: thoughtContent, Thought: true}}, aggregatedContent.Parts...)
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
func (r *ResponsesModel) handleTextDelta(
	data string,
	thinkParser *thinkTagStreamParser,
	textContent *string,
	thoughtContent *string,
	yield func(*model.LLMResponse, error) bool,
) bool {
	var delta ResponsesTextDelta
	if err := json.Unmarshal([]byte(data), &delta); err != nil {
		respLog.Warn("解析文本增量失败: %v", err)
		return true
	}
	return r.emitTextSegments(thinkParser.Feed(delta.Delta), textContent, thoughtContent, yield)
}

func (r *ResponsesModel) emitTextSegments(
	segments []thinkSegment,
	textContent *string,
	thoughtContent *string,
	yield func(*model.LLMResponse, error) bool,
) bool {
	for _, seg := range segments {
		if seg.Text == "" {
			continue
		}
		if seg.Thought {
			*thoughtContent += seg.Text
		} else {
			*textContent += seg.Text
		}
		part := &genai.Part{Text: seg.Text, Thought: seg.Thought}
		llmResp := &model.LLMResponse{
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{part}},
			Partial:      true,
			TurnComplete: false,
		}
		if !yield(llmResp, nil) {
			return false
		}
	}
	return true
}

// handleFuncArgsDelta 处理函数调用参数增量事件
func (r *ResponsesModel) handleFuncArgsDelta(data string, toolCallsMap map[string]*responsesToolCallBuilder) {
	var delta ResponsesFuncCallArgsDelta
	if err := json.Unmarshal([]byte(data), &delta); err != nil {
		respLog.Warn("解析函数参数增量失败: %v", err)
		return
	}
	if builder, exists := toolCallsMap[delta.ItemID]; exists {
		builder.args += delta.Delta
	}
}

// handleOutputItemAdded 处理 output item added 事件
func (r *ResponsesModel) handleOutputItemAdded(data string, toolCallsMap map[string]*responsesToolCallBuilder, toolCallOrder *[]string) {
	var added ResponsesOutputItemAdded
	if err := json.Unmarshal([]byte(data), &added); err != nil {
		respLog.Warn("解析输出项添加事件失败: %v", err)
		return
	}
	if added.Item.Type == "function_call" {
		toolCallsMap[added.Item.ID] = &responsesToolCallBuilder{
			itemID: added.Item.ID,
			callID: added.Item.CallID,
			name:   added.Item.Name,
		}
		*toolCallOrder = append(*toolCallOrder, added.Item.ID)
	}
}

// handleOutputItemDone 处理 output item done 事件
func (r *ResponsesModel) handleOutputItemDone(data string, toolCallsMap map[string]*responsesToolCallBuilder, toolCallOrder *[]string) {
	var done ResponsesOutputItemDone
	if err := json.Unmarshal([]byte(data), &done); err != nil {
		respLog.Warn("解析输出项完成事件失败: %v", err)
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
			*toolCallOrder = append(*toolCallOrder, done.Item.ID)
		}
	}
}

// handleCompleted 处理 response.completed 事件
func (r *ResponsesModel) handleCompleted(data string, usageMetadata **genai.GenerateContentResponseUsageMetadata) {
	var completed ResponsesCompleted
	if err := json.Unmarshal([]byte(data), &completed); err != nil {
		respLog.Warn("解析完成事件失败: %v", err)
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
