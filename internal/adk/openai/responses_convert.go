package openai

import (
	"encoding/json"
	"fmt"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// toResponsesRequest 将 ADK 请求转换为 Responses API 请求
func toResponsesRequest(req *model.LLMRequest, modelName string) (CreateResponseRequest, error) {
	// 转换 input 消息
	inputItems, err := toResponsesInputItems(req.Contents)
	if err != nil {
		return CreateResponseRequest{}, err
	}

	apiReq := CreateResponseRequest{
		Model: modelName,
		Input: inputItems,
	}

	if req.Config == nil {
		return apiReq, nil
	}

	// 提取系统指令到顶层 instructions 字段
	if req.Config.SystemInstruction != nil {
		apiReq.Instructions = extractTextFromContent(req.Config.SystemInstruction)
	}

	// 处理 thinking/reasoning 配置
	if req.Config.ThinkingConfig != nil {
		reasoning := &ResponsesReasoning{}
		switch req.Config.ThinkingConfig.ThinkingLevel {
		case genai.ThinkingLevelLow:
			reasoning.Effort = "low"
		case genai.ThinkingLevelHigh:
			reasoning.Effort = "high"
		default:
			reasoning.Effort = "medium"
		}
		apiReq.Reasoning = reasoning
	}

	// 转换工具定义
	if len(req.Config.Tools) > 0 {
		apiReq.Tools = convertResponsesTools(req.Config.Tools)
	}

	// 应用生成参数
	if req.Config.Temperature != nil {
		t := float32(*req.Config.Temperature)
		apiReq.Temperature = &t
	}
	if req.Config.MaxOutputTokens > 0 {
		apiReq.MaxOutputTokens = int(req.Config.MaxOutputTokens)
	}
	if req.Config.TopP != nil {
		p := float32(*req.Config.TopP)
		apiReq.TopP = &p
	}
	if len(req.Config.StopSequences) > 0 {
		apiReq.Stop = req.Config.StopSequences
	}

	return apiReq, nil
}

// toResponsesInputItems 将 genai.Content 列表转换为 Responses API input
func toResponsesInputItems(contents []*genai.Content) ([]ResponsesInputItem, error) {
	var items []ResponsesInputItem

	for _, content := range contents {
		newItems, err := toResponsesInputItem(content)
		if err != nil {
			return nil, err
		}
		items = append(items, newItems...)
	}

	return items, nil
}

// toResponsesInputItem 将单个 genai.Content 转换为 Responses API input 项
func toResponsesInputItem(content *genai.Content) ([]ResponsesInputItem, error) {
	var items []ResponsesInputItem

	// 先处理 function response（工具调用结果）
	for _, part := range content.Parts {
		if part.FunctionResponse != nil {
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("序列化函数响应失败: %w", err)
			}
			items = append(items, ResponsesInputItem{
				Type:   "function_call_output",
				CallID: part.FunctionResponse.ID,
				Output: string(responseJSON),
			})
		}
	}

	// 收集文本、reasoning、函数调用
	var textContent string
	var toolCallItems []ResponsesInputItem

	for _, part := range content.Parts {
		if part.FunctionResponse != nil {
			continue // 已处理
		}
		if part.Text != "" && !part.Thought {
			textContent += part.Text
		}
		if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("序列化函数参数失败: %w", err)
			}
			toolCallItems = append(toolCallItems, ResponsesInputItem{
				Type:      "function_call",
				ID:        part.FunctionCall.ID,
				CallID:    part.FunctionCall.ID,
				Name:      part.FunctionCall.Name,
				Arguments: string(argsJSON),
			})
		}
	}

	// 构建普通消息
	role := convertRoleForResponses(content.Role)
	if textContent != "" {
		items = append(items, ResponsesInputItem{
			Role:    role,
			Content: textContent,
		})
	}

	// assistant 的工具调用作为独立 input 项
	items = append(items, toolCallItems...)

	return items, nil
}

// convertRoleForResponses 转换角色为 Responses API 格式
func convertRoleForResponses(role string) string {
	switch role {
	case "user":
		return "user"
	case "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}

// convertResponsesTools 转换工具定义为 Responses API 扁平化格式
func convertResponsesTools(genaiTools []*genai.Tool) []ResponsesTool {
	var tools []ResponsesTool
	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}
		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}
			tools = append(tools, ResponsesTool{
				Type:        "function",
				Name:        funcDecl.Name,
				Description: funcDecl.Description,
				Parameters:  params,
			})
		}
	}
	return tools
}

// convertResponsesResponse 将 Responses API 响应转换为 ADK LLMResponse
func convertResponsesResponse(resp *CreateResponseResponse) (*model.LLMResponse, error) {
	if len(resp.Output) == 0 {
		return nil, ErrNoChoicesInResponse
	}

	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					content.Parts = append(content.Parts, &genai.Part{Text: part.Text})
				case "reasoning":
					content.Parts = append(content.Parts, &genai.Part{
						Text:    part.Text,
						Thought: true,
					})
				}
			}
		case "function_call":
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   item.CallID,
					Name: item.Name,
					Args: parseJSONArgs(item.Arguments),
				},
			})
		}
	}

	// 处理 usage
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if resp.Usage != nil {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.InputTokens),
			CandidatesTokenCount: int32(resp.Usage.OutputTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  genai.FinishReasonStop,
		TurnComplete:  true,
	}, nil
}
