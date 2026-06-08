package api

// CommandCode API types (internal)

type CCToolOutput struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

type CCContentPart struct {
	Type       string        `json:"type"`
	Text       *string       `json:"text,omitempty"`
	Image      any           `json:"image,omitempty"`
	MediaType  *string       `json:"mediaType,omitempty"`
	ID         *string       `json:"id,omitempty"`
	Name       *string       `json:"name,omitempty"`
	Input      any           `json:"input,omitempty"`
	ToolCallID *string       `json:"toolCallId,omitempty"`
	ToolName   *string       `json:"toolName,omitempty"`
	Output     *CCToolOutput `json:"output,omitempty"`
	ToolUseID  *string       `json:"tool_use_id,omitempty"`
	Content    any           `json:"content,omitempty"`
}

type CCMessage struct {
	Role    string          `json:"role"`
	Content []CCContentPart `json:"content"`
}

type CCChatParams struct {
	Model           string      `json:"model"`
	Messages        []CCMessage `json:"messages"`
	Tools           []any       `json:"tools"`
	System          string      `json:"system"`
	MaxTokens       int         `json:"max_tokens"`
	Temperature     float64     `json:"temperature"`
	Stream          bool        `json:"stream"`
	ReasoningEffort string      `json:"reasoning_effort,omitempty"`
}

type CCConfig struct {
	WorkingDir    string   `json:"workingDir"`
	Date          string   `json:"date"`
	Environment   string   `json:"environment"`
	Structure     []string `json:"structure"`
	IsGitRepo     bool     `json:"isGitRepo"`
	CurrentBranch string   `json:"currentBranch"`
	MainBranch    string   `json:"mainBranch"`
	GitStatus     string   `json:"gitStatus"`
	RecentCommits []string `json:"recentCommits"`
}

type CCRequestBody struct {
	Config   CCConfig     `json:"config"`
	Memory   string       `json:"memory"`
	Taste    string       `json:"taste"`
	Skills   string       `json:"skills"`
	Params   CCChatParams `json:"params"`
	ThreadID string       `json:"threadId"`
}

// CCUsageInputDetails breaks down the prompt tokens. noCacheTokens +
// cacheReadTokens == inputTokens; cacheReadTokens is the subset served from the
// provider prompt cache.
type CCUsageInputDetails struct {
	NoCacheTokens    int `json:"noCacheTokens"`
	CacheReadTokens  int `json:"cacheReadTokens"`
	CacheWriteTokens int `json:"cacheWriteTokens"`
}

// CCUsageOutputDetails breaks down the completion tokens into visible text and
// internal reasoning tokens.
type CCUsageOutputDetails struct {
	TextTokens      int `json:"textTokens"`
	ReasoningTokens int `json:"reasoningTokens"`
}

// CCUsage is the rich usage object CommandCode emits on "finish-step" (as
// "usage") and "finish" (as "totalUsage"). It carries cache and reasoning token
// breakdowns that the proxy maps onto the OpenAI usage details.
type CCUsage struct {
	InputTokens        int                  `json:"inputTokens"`
	OutputTokens       int                  `json:"outputTokens"`
	TotalTokens        int                  `json:"totalTokens"`
	ReasoningTokens    int                  `json:"reasoningTokens"`
	CachedInputTokens  int                  `json:"cachedInputTokens"`
	InputTokenDetails  CCUsageInputDetails  `json:"inputTokenDetails"`
	OutputTokenDetails CCUsageOutputDetails `json:"outputTokenDetails"`
}

// CachedTokens returns the number of prompt tokens served from cache, tolerating
// either the top-level cachedInputTokens or the nested cacheReadTokens.
func (u *CCUsage) CachedTokens() int {
	if u == nil {
		return 0
	}
	if u.CachedInputTokens > 0 {
		return u.CachedInputTokens
	}
	return u.InputTokenDetails.CacheReadTokens
}

// Reasoning returns the number of reasoning tokens, tolerating either the
// top-level reasoningTokens or the nested outputTokenDetails.reasoningTokens.
func (u *CCUsage) Reasoning() int {
	if u == nil {
		return 0
	}
	if u.ReasoningTokens > 0 {
		return u.ReasoningTokens
	}
	return u.OutputTokenDetails.ReasoningTokens
}

// Total returns the total token count, falling back to input+output when the
// upstream omits an explicit total.
func (u *CCUsage) Total() int {
	if u == nil {
		return 0
	}
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.InputTokens + u.OutputTokens
}

// CCError is the structured error payload carried by an "error" stream event.
type CCError struct {
	Type        string `json:"type"`
	Message     string `json:"message"`
	StatusCode  *int   `json:"statusCode"`
	IsRetryable bool   `json:"isRetryable"`
}

// CCGatewayRouting describes which upstream provider actually served the
// request after CommandCode's gateway routing/fallback logic.
type CCGatewayRouting struct {
	ResolvedProvider string `json:"resolvedProvider"`
	FinalProvider    string `json:"finalProvider"`
	CanonicalSlug    string `json:"canonicalSlug"`
}

// CCGateway carries gateway-level metadata: cost (as decimal strings) and the
// generation id, plus routing details.
type CCGateway struct {
	Cost         string            `json:"cost"`
	MarketCost   string            `json:"marketCost"`
	GatewayCost  string            `json:"gatewayCost"`
	GenerationID string            `json:"generationId"`
	Routing      *CCGatewayRouting `json:"routing"`
}

// CCProviderMetadata wraps the gateway metadata block attached to "finish-step"
// and emitted as a standalone "provider-metadata" event.
type CCProviderMetadata struct {
	Gateway *CCGateway `json:"gateway"`
}

type CCStreamEvent struct {
	Type             string              `json:"type"`
	Text             string              `json:"text"`
	ID               string              `json:"id"`
	Delta            string              `json:"delta"`
	Input            map[string]any      `json:"input"`
	ToolCallID       string              `json:"toolCallId"`
	ToolName         string              `json:"toolName"`
	FinishReason     string              `json:"finishReason"`
	Error            *CCError            `json:"error"`
	TotalUsage       *CCUsage            `json:"totalUsage"`
	Usage            *CCUsage            `json:"usage"`
	ProviderMetadata *CCProviderMetadata `json:"providerMetadata"`
}
