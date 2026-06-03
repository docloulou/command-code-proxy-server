package proxy

import "strings"

// Reasoning effort values supported per canonical model, mirroring the
// CommandCode model catalog. An entry with an empty slice means the model has
// reasoning but does not expose a selectable effort level (so we drop the
// effort to avoid upstream errors). Models absent from the map are unknown to
// the proxy and the requested effort is passed through unchanged.
var modelReasoningEfforts = map[string][]string{
	"deepseek/deepseek-v4-pro":     {"high", "max"},
	"deepseek/deepseek-v4-flash":   {"high", "max"},
	"google/gemini-3.1-flash-lite": {"low", "medium", "high"},
	"Qwen/Qwen3.6-Max-Preview":     {},
	"Qwen/Qwen3.6-Plus":            {},
	"stepfun/Step-3.5-Flash":       {},
	"moonshotai/Kimi-K2.6":         {},
	"moonshotai/Kimi-K2.5":         {},
	"zai-org/GLM-5.1":              {},
	"zai-org/GLM-5":                {},
	"MiniMaxAI/MiniMax-M2.7":       {},
	"MiniMaxAI/MiniMax-M2.5":       {},
}

// Global ordering of reasoning effort levels from lightest to heaviest, used to
// clamp a requested effort to the nearest value a model actually supports.
var effortRank = map[string]int{
	"minimal": 0,
	"low":     1,
	"medium":  2,
	"high":    3,
	"xhigh":   4,
	"max":     5,
}

// ResolveReasoningEffort maps a client-requested reasoning_effort to a value the
// target model accepts. It returns an empty string when nothing should be sent
// upstream (no request, or the model has no selectable effort).
func ResolveReasoningEffort(model, requested string) string {
	requested = strings.ToLower(strings.TrimSpace(requested))
	if requested == "" {
		return ""
	}

	supported, known := modelReasoningEfforts[model]
	if !known {
		return requested // unknown model: let the upstream decide
	}
	if len(supported) == 0 {
		return "" // model has reasoning but no selectable effort
	}

	for _, s := range supported {
		if s == requested {
			return requested
		}
	}

	reqRank, ok := effortRank[requested]
	if !ok {
		return supported[len(supported)-1] // unknown alias: pick the strongest
	}

	best := supported[0]
	bestDist := absInt(effortRank[best] - reqRank)
	for _, s := range supported[1:] {
		d := absInt(effortRank[s] - reqRank)
		if d < bestDist || (d == bestDist && effortRank[s] > effortRank[best]) {
			best = s
			bestDist = d
		}
	}
	return best
}

func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// Map model name if client sends short name
func MapModel(name string) string {
	switch strings.ToLower(name) {
	case "deepseek-v4-pro", "deepseek-v4", "deepseek-pro":
		return "deepseek/deepseek-v4-pro"
	case "deepseek-v4-flash", "deepseek-flash":
		return "deepseek/deepseek-v4-flash"
	case "minimax-m2.7", "minimax2.7":
		return "MiniMaxAI/MiniMax-M2.7"
	case "minimax-m2.5", "minimax2.5", "minimax":
		return "MiniMaxAI/MiniMax-M2.5"
	case "glm-5.1":
		return "zai-org/GLM-5.1"
	case "glm-5":
		return "zai-org/GLM-5"
	case "kimi-k2.6", "kimi2.6":
		return "moonshotai/Kimi-K2.6"
	case "kimi-k2.5", "kimi2.5":
		return "moonshotai/Kimi-K2.5"
	case "qwen-3.6-max-preview", "qwen3.6-max":
		return "Qwen/Qwen3.6-Max-Preview"
	case "qwen-3.6-plus", "qwen3.6-plus", "qwen3.6":
		return "Qwen/Qwen3.6-Plus"
	case "step-3.5-flash", "step3.5":
		return "stepfun/Step-3.5-Flash"
	case "gemini-3.1-flash-lite", "gemini-flash-lite":
		return "google/gemini-3.1-flash-lite"
	default:
		return name // pass through as-is
	}
}
