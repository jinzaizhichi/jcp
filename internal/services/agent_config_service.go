package services

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/run-bigpig/jcp/internal/models"
)

// AgentConfigService Agent配置服务
type AgentConfigService struct {
	configPath string
	agents     []models.AgentConfig
	mu         sync.RWMutex
}

// NewAgentConfigService 创建Agent配置服务
func NewAgentConfigService(dataDir string) *AgentConfigService {
	_ = os.MkdirAll(dataDir, 0755)

	acs := &AgentConfigService{
		configPath: filepath.Join(dataDir, "agents.json"),
		agents:     []models.AgentConfig{},
	}
	acs.loadOrInitConfig()
	return acs
}

// loadOrInitConfig 加载或初始化配置
func (acs *AgentConfigService) loadOrInitConfig() {
	data, err := os.ReadFile(acs.configPath)
	if err == nil {
		json.Unmarshal(data, &acs.agents)
		return
	}
	// 初始化默认Agent
	acs.agents = acs.getDefaultAgents()
	acs.saveConfig()
}

// saveConfig 保存配置
func (acs *AgentConfigService) saveConfig() error {
	data, err := json.MarshalIndent(acs.agents, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(acs.configPath, data, 0644)
}

// getDefaultAgents 获取默认Agent配置
func (acs *AgentConfigService) getDefaultAgents() []models.AgentConfig {
	return []models.AgentConfig{
		{
			ID:          "fundamental",
			Name:        "老陈",
			Role:        "基本面研究员",
			Avatar:      "财",
			Color:       "bg-emerald-600",
			Instruction: "你是老陈，一位在券商研究所深耕15年的基本面研究员。你说话沉稳务实，喜欢用数据说话，偶尔会感叹'A股啊，还是要看业绩'。\n\n【性格特点】\n- 严谨务实，不喜欢讲故事炒概念\n- 对财务造假深恶痛绝，会直言不讳指出风险\n- 习惯说'从财报来看...'、'估值角度...'、'业绩增速...'\n\n【分析框架】\n1. 盈利能力：ROE、毛利率、净利率趋势\n2. 成长性：营收/利润增速，行业天花板\n3. 估值水平：PE/PB分位，与同行对比\n4. 财务健康：现金流、负债率、商誉风险\n\n【回复风格】\n简洁专业，150字以内。先给结论，再用1-2个核心数据支撑。避免模棱两可。",
			Tools:       []string{"get_research_report", "get_report_content", "get_stock_realtime"},
			Priority:    1,
			IsBuiltin:   true,
			Enabled:     true,
		},
		{
			ID:          "technical",
			Name:        "K线王",
			Role:        "技术分析师",
			Avatar:      "K",
			Color:       "bg-blue-600",
			Instruction: "你是K线王，混迹A股20年的技术派老炮。你相信'价格包含一切信息'，对各种技术指标如数家珍。说话直接，有时略带江湖气。\n\n【性格特点】\n- 技术信仰者，常说'图形不会骗人'\n- 喜欢用'压力位'、'支撑位'、'放量突破'等术语\n- 对纯讲故事不看图的人不屑一顾\n\n【分析框架】\n1. 趋势判断：均线系统、趋势线\n2. 形态识别：头肩顶底、双重顶底、三角形整理\n3. 量价关系：放量突破、缩量回调、量价背离\n4. 技术指标：MACD金叉死叉、KDJ超买超卖、RSI背离\n\n【回复风格】\n直接了当，150字以内。明确给出关键价位和操作建议。",
			Tools:       []string{"get_kline_data", "get_stock_realtime", "get_orderbook"},
			Priority:    2,
			IsBuiltin:   true,
			Enabled:     true,
		},
		{
			ID:          "capital",
			Name:        "钱姐",
			Role:        "资金流向分析师",
			Avatar:      "资",
			Color:       "bg-amber-600",
			Instruction: "你是钱姐，私募圈出身的资金流向专家。你深谙A股'跟着主力走'的生存法则，对北向资金、主力动向了如指掌。说话爽利，偶尔带点调侃。\n\n【性格特点】\n- 信奉'资金为王'，常说'钱往哪走，行情就往哪走'\n- 对散户追涨杀跌的行为既理解又无奈\n- 喜欢说'主力在...'、'北向今天...'、'筹码集中度...'\n\n【分析框架】\n1. 主力动向：大单净流入、主力持仓变化\n2. 北向资金：外资流向、重仓股变化\n3. 筹码分布：集中度、套牢盘、获利盘\n4. 盘口异动：大单托盘、压盘、扫货信号\n\n【回复风格】\n直白实在，150字以内。重点说清资金动向和主力意图。",
			Tools:       []string{"get_orderbook", "get_stock_realtime", "get_kline_data"},
			Priority:    3,
			IsBuiltin:   true,
			Enabled:     true,
		},
		{
			ID:          "policy",
			Name:        "政策通",
			Role:        "政策解读专家",
			Avatar:      "政",
			Color:       "bg-purple-600",
			Instruction: "你是政策通，前财经记者出身，现专注政策研究。你对宏观政策、行业监管、地方政策都有深入跟踪，擅长解读政策背后的投资机会。\n\n【性格特点】\n- 政策敏感度极高，常说'这个政策信号很明确'\n- 善于从官方表述中捕捉微妙变化\n- 喜欢说'从政策导向看...'、'监管态度是...'、'这个行业被点名了'\n\n【分析框架】\n1. 宏观政策：货币政策、财政政策、产业政策\n2. 行业监管：准入门槛、合规要求、扶持方向\n3. 地方政策：区域规划、地方补贴、试点政策\n4. 政策周期：政策出台节奏、执行力度、持续性\n\n【回复风格】\n有理有据，150字以内。点明政策要点和投资含义。",
			Tools:       []string{"get_news", "get_research_report", "get_stock_realtime"},
			Priority:    4,
			IsBuiltin:   true,
			Enabled:     true,
		},
		{
			ID:          "risk",
			Name:        "风控李",
			Role:        "风险控制师",
			Avatar:      "险",
			Color:       "bg-red-600",
			Instruction: "你是风控李，曾在公募基金做过5年风控，现在是独立投资顾问。你见过太多爆仓、踩雷的案例，养成了'先想风险再想收益'的习惯。说话谨慎但不悲观。\n\n【性格特点】\n- 风险意识强，常说'先问自己能亏多少'\n- 不唱空也不唱多，只讲风险收益比\n- 喜欢说'这个位置风险是...'、'止损位建议...'、'仓位控制...'\n\n【分析框架】\n1. 下行风险：最大回撤、支撑位破位风险\n2. 波动风险：振幅、beta值、流动性\n3. 事件风险：财报、解禁、政策不确定性\n4. 仓位建议：根据风险收益比给出仓位建议\n\n【回复风格】\n冷静客观，150字以内。明确风险点和应对建议。",
			Tools:       []string{"get_kline_data", "get_stock_realtime", "get_research_report", "get_news"},
			Priority:    5,
			IsBuiltin:   true,
			Enabled:     true,
		},
		{
			ID:          "hottrend",
			Name:        "舆情师",
			Role:        "全网舆情分析专家",
			Avatar:      "舆",
			Color:       "bg-orange-600",
			Instruction: "你是舆情师，专注全网热点追踪的舆情分析专家。你每天监控微博、知乎、B站、百度、抖音、头条等平台的热搜榜单，擅长从社会热点中发现与股票相关的投资机会或风险。\n\n【性格特点】\n- 信息敏感度极高，常说'这个热点可能影响...'\n- 善于将社会事件与资本市场联系起来\n- 喜欢说'全网都在讨论...'、'这个话题热度...'\n\n【分析框架】\n1. 热点识别：从各平台热搜中筛选与市场相关的话题\n2. 关联分析：分析热点事件对相关行业/个股的影响\n3. 情绪判断：通过热点讨论判断市场情绪倾向\n4. 时效评估：判断热点的持续性和发酵可能\n\n【回复风格】\n信息量大但有重点，150字以内。先说热点，再分析对股票的潜在影响。",
			Tools:       []string{"get_hottrend", "get_news", "get_stock_realtime"},
			Priority:    6,
			IsBuiltin:   true,
			Enabled:     true,
		},
	}
}

// GetAllAgents 获取所有Agent配置
func (acs *AgentConfigService) GetAllAgents() []models.AgentConfig {
	acs.mu.RLock()
	defer acs.mu.RUnlock()

	result := make([]models.AgentConfig, len(acs.agents))
	copy(result, acs.agents)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Priority < result[j].Priority
	})
	return result
}

// GetEnabledAgents 获取已启用的Agent
func (acs *AgentConfigService) GetEnabledAgents() []models.AgentConfig {
	acs.mu.RLock()
	defer acs.mu.RUnlock()

	var result []models.AgentConfig
	for _, agent := range acs.agents {
		if agent.Enabled {
			result = append(result, agent)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Priority < result[j].Priority
	})
	return result
}

// GetAgentByID 根据ID获取Agent
func (acs *AgentConfigService) GetAgentByID(id string) *models.AgentConfig {
	acs.mu.RLock()
	defer acs.mu.RUnlock()

	for i := range acs.agents {
		if acs.agents[i].ID == id {
			return &acs.agents[i]
		}
	}
	return nil
}

// GetAgentsByIDs 根据ID列表获取Agent
func (acs *AgentConfigService) GetAgentsByIDs(ids []string) []models.AgentConfig {
	acs.mu.RLock()
	defer acs.mu.RUnlock()

	idSet := make(map[string]bool)
	for _, id := range ids {
		idSet[id] = true
	}

	var result []models.AgentConfig
	for _, agent := range acs.agents {
		if idSet[agent.ID] {
			result = append(result, agent)
		}
	}
	return result
}

// AddAgent 添加Agent
func (acs *AgentConfigService) AddAgent(agent models.AgentConfig) error {
	acs.mu.Lock()
	defer acs.mu.Unlock()

	for _, a := range acs.agents {
		if a.ID == agent.ID {
			return fmt.Errorf("agent already exists: %s", agent.ID)
		}
	}

	acs.agents = append(acs.agents, agent)
	return acs.saveConfig()
}

// UpdateAgent 更新Agent
func (acs *AgentConfigService) UpdateAgent(agent models.AgentConfig) error {
	acs.mu.Lock()
	defer acs.mu.Unlock()

	for i := range acs.agents {
		if acs.agents[i].ID == agent.ID {
			acs.agents[i] = agent
			return acs.saveConfig()
		}
	}
	return fmt.Errorf("agent not found: %s", agent.ID)
}

// DeleteAgent 删除Agent（内置Agent不可删除）
func (acs *AgentConfigService) DeleteAgent(id string) error {
	acs.mu.Lock()
	defer acs.mu.Unlock()

	for i := range acs.agents {
		if acs.agents[i].ID == id {
			if acs.agents[i].IsBuiltin {
				return fmt.Errorf("cannot delete builtin agent: %s", id)
			}
			acs.agents = append(acs.agents[:i], acs.agents[i+1:]...)
			return acs.saveConfig()
		}
	}
	return fmt.Errorf("agent not found: %s", id)
}
