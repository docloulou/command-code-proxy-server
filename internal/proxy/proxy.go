package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/dev2k6/command-code-proxy-server/internal/api"
	"github.com/dev2k6/command-code-proxy-server/internal/models"
	"github.com/dev2k6/command-code-proxy-server/internal/version"
	"github.com/google/uuid"
)

const defaultBaseURL = "https://api.commandcode.ai"
const debugLogLimit = 20000

// Upstream resiliency tuning. The HTTP client intentionally has no overall
// timeout so that long-lived streaming responses are not cut off mid-flight;
// instead, per-phase transport timeouts and the inbound request context bound
// the request lifetime.
const (
	dialTimeout           = 30 * time.Second
	responseHeaderTimeout = 120 * time.Second
	idleConnTimeout       = 90 * time.Second
	tlsHandshakeTimeout   = 10 * time.Second

	maxUpstreamRetries = 2
	retryBaseDelay     = 400 * time.Millisecond
	retryMaxDelay      = 4 * time.Second
)

func truncateLog(s string) string {
	if len(s) <= debugLogLimit {
		return s
	}
	return s[:debugLogLimit] + fmt.Sprintf("... [truncated %d bytes]", len(s)-debugLogLimit)
}

func (p *Proxy) debugf(format string, args ...any) {
	if p.Debug {
		log.Printf(format, args...)
	}
}

func (p *Proxy) writeOpenAIError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(api.OpenAIErrorResponse{Error: api.OpenAIError{
		Message: message,
		Type:    errType,
		Param:   nil,
		Code:    nil,
	}})
}

func normalizeFinishReason(reason string) string {
	switch reason {
	case "tool_calls", "tool-calls":
		return "tool_calls"
	case "length", "max_tokens":
		return "length"
	case "content_filter", "content-filter":
		return "content_filter"
	default:
		return "stop"
	}
}

// Proxy struct
type Proxy struct {
	APIKey  string
	BaseURL string
	Client  *http.Client
	Debug   bool
}

// NewProxy creates a new proxy instance
func NewProxy(apiKey string) *Proxy {
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   dialTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       idleConnTimeout,
		TLSHandshakeTimeout:   tlsHandshakeTimeout,
		ResponseHeaderTimeout: responseHeaderTimeout,
		ExpectContinueTimeout: 1 * time.Second,
	}

	return &Proxy{
		APIKey:  apiKey,
		BaseURL: defaultBaseURL,
		// No Client.Timeout on purpose: streaming responses must be allowed to
		// outlive any fixed deadline. Cancellation is driven by the request
		// context; stalls before the first byte are bounded by the transport's
		// dial and response-header timeouts.
		Client: &http.Client{Transport: transport},
	}
}

// BuildRequest builds the CommandCode request body
func (p *Proxy) BuildRequest(openAIReq api.OpenAIChatRequest) (api.CCRequestBody, error) {
	model := MapModel(openAIReq.Model)
	system, msgs := ExtractSystem(openAIReq.Messages)
	ccMessages := ConvertMessages(msgs)

	temperature := 0.3
	maxTokens := 64000
	if openAIReq.Temperature != nil {
		temperature = *openAIReq.Temperature
	}
	if openAIReq.MaxTokens != nil {
		maxTokens = *openAIReq.MaxTokens
	}
	if openAIReq.MaxCompletionTokens != nil {
		maxTokens = *openAIReq.MaxCompletionTokens
	}

	tools := ConvertTools(openAIReq.Tools)

	ccBody := api.CCRequestBody{
		Config: api.CCConfig{
			WorkingDir:    ".",
			Date:          time.Now().Format("2006-01-02"),
			Environment:   "cli",
			Structure:     []string{},
			IsGitRepo:     false,
			CurrentBranch: "",
			MainBranch:    "main",
			GitStatus:     "",
			RecentCommits: []string{},
		},
		Memory: "",
		Taste:  "",
		Skills: "",
		Params: api.CCChatParams{
			Model:           model,
			Messages:        ccMessages,
			Tools:           tools,
			System:          system,
			MaxTokens:       maxTokens,
			Temperature:     temperature,
			Stream:          true,
			ReasoningEffort: ResolveReasoningEffort(model, openAIReq.ReasoningEffort),
		},
		ThreadID: uuid.New().String(),
	}

	return ccBody, nil
}

// CreateUpstreamRequest creates a new HTTP request to the CommandCode API
func (p *Proxy) CreateUpstreamRequest(ctx context.Context, ccBody api.CCRequestBody, apiKey string) (*http.Request, error) {
	reqJSON, err := json.Marshal(ccBody)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}

	p.debugf("[DEBUG] CommandCode request body: %s", truncateLog(string(reqJSON)))

	ccReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.BaseURL+"/alpha/generate", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create upstream request: %w", err)
	}

	ccReq.Header.Set("Content-Type", "application/json")
	ccReq.Header.Set("Authorization", "Bearer "+apiKey)
	ccVersion := version.GetCommandCodeVersion()
	if envCC := os.Getenv("CC_VERSION"); envCC != "" {
		ccVersion = envCC
	}
	ccReq.Header.Set("x-command-code-version", ccVersion)
	ccReq.Header.Set("x-cli-environment", "production")
	ccReq.Header.Set("Accept", "text/event-stream")

	return ccReq, nil
}

// CallUpstream makes the request to CommandCode API
func (p *Proxy) CallUpstream(req *http.Request) (*http.Response, error) {
	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream error: %w", err)
	}
	return resp, nil
}

// CallUpstreamWithRetry sends the request and retries transient failures
// (network errors and 429/5xx-class statuses) with exponential backoff. It is
// safe because retries happen before any bytes are streamed to the client and
// the request body is rebuilt from the in-memory CC payload on each attempt.
func (p *Proxy) CallUpstreamWithRetry(ctx context.Context, ccBody api.CCRequestBody, apiKey string) (*http.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= maxUpstreamRetries; attempt++ {
		if attempt > 0 {
			delay := backoffDelay(attempt)
			p.debugf("[DEBUG] retrying upstream (attempt %d/%d) after %s: %v", attempt, maxUpstreamRetries, delay, lastErr)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		req, err := p.CreateUpstreamRequest(ctx, ccBody, apiKey)
		if err != nil {
			return nil, err
		}

		resp, err := p.Client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			lastErr = fmt.Errorf("upstream error: %w", err)
			continue
		}

		if attempt < maxUpstreamRetries && shouldRetryStatus(resp.StatusCode) {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			lastErr = fmt.Errorf("upstream returned status %d", resp.StatusCode)
			continue
		}

		return resp, nil
	}

	if lastErr == nil {
		lastErr = errors.New("upstream request failed")
	}
	return nil, lastErr
}

// shouldRetryStatus reports whether an HTTP status code is worth retrying.
func shouldRetryStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, // 429
		http.StatusBadGateway,         // 502
		http.StatusServiceUnavailable, // 503
		http.StatusGatewayTimeout:     // 504
		return true
	default:
		return false
	}
}

// backoffDelay returns an exponential backoff capped at retryMaxDelay.
func backoffDelay(attempt int) time.Duration {
	delay := retryBaseDelay << (attempt - 1)
	if delay > retryMaxDelay {
		return retryMaxDelay
	}
	return delay
}

// HandleChatCompletions handles the /v1/chat/completions endpoint
func (p *Proxy) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		p.writeOpenAIError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	// Get API key from client Authorization header or server default
	apiKey := r.Header.Get("Authorization")
	if apiKey != "" {
		apiKey = strings.TrimPrefix(apiKey, "Bearer ")
		apiKey = strings.TrimSpace(apiKey)
	} else if p.APIKey != "" {
		apiKey = p.APIKey
	} else {
		p.writeOpenAIError(w, http.StatusUnauthorized, "API key required. Set Authorization header.", "authentication_error")
		return
	}

	// Read request
	body, err := io.ReadAll(r.Body)
	if err != nil {
		p.writeOpenAIError(w, http.StatusBadRequest, "Failed to read body", "invalid_request_error")
		return
	}

	p.debugf("[DEBUG] Client request body: %s", truncateLog(string(body)))

	var openAIReq api.OpenAIChatRequest
	if err := json.Unmarshal(body, &openAIReq); err != nil {
		p.writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("Invalid JSON: %s", err.Error()), "invalid_request_error")
		return
	}

	if len(openAIReq.Messages) == 0 {
		p.writeOpenAIError(w, http.StatusBadRequest, "messages array is required", "invalid_request_error")
		return
	}

	// Build CommandCode request
	ccBody, err := p.BuildRequest(openAIReq)
	if err != nil {
		p.writeOpenAIError(w, http.StatusInternalServerError, "Failed to build request", "server_error")
		return
	}

	// Call upstream with retry/backoff for transient failures.
	ccResp, err := p.CallUpstreamWithRetry(r.Context(), ccBody, apiKey)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		p.writeOpenAIError(w, http.StatusBadGateway, err.Error(), "api_error")
		return
	}
	defer ccResp.Body.Close()

	if ccResp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(ccResp.Body)
		message := fmt.Sprintf("Upstream error: %s", string(errBody))
		log.Printf("[ERROR] Upstream returned %d: %s", ccResp.StatusCode, string(errBody))
		status := http.StatusBadGateway
		if ccResp.StatusCode >= http.StatusBadRequest && ccResp.StatusCode < http.StatusInternalServerError {
			status = ccResp.StatusCode
		}
		p.writeOpenAIError(w, status, message, "api_error")
		return
	}

	requestID := "chatcmpl-" + uuid.New().String()[:29]
	created := time.Now().Unix()

	if openAIReq.Stream {
		includeUsage := openAIReq.StreamOptions != nil && openAIReq.StreamOptions.IncludeUsage
		p.StreamResponse(w, r, ccResp, requestID, ccBody.Params.Model, created, includeUsage)
	} else {
		p.NonStreamResponse(w, ccResp, requestID, ccBody.Params.Model, created)
	}
}

// StreamResponse handles streaming response from CommandCode to OpenAI SSE
func (p *Proxy) StreamResponse(w http.ResponseWriter, r *http.Request, ccResp *http.Response, requestID, model string, created int64, includeUsage bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		p.writeOpenAIError(w, http.StatusInternalServerError, "Streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	sentRole := false
	toolCallIndex := 0
	toolCallIndexes := map[string]int{}
	var usage *api.OpenAIUsage
	finished := false

	readErr := forEachLine(ccResp.Body, func(raw []byte) bool {
		if r.Context().Err() != nil {
			return false
		}

		event, ok := parseCCEvent(raw)
		if !ok {
			return true
		}
		p.debugf("[DEBUG] CommandCode stream line: %s", truncateLog(string(bytes.TrimSpace(raw))))

		switch event.Type {
		case "text-delta":
			delta := api.OpenAIDelta{Content: event.Text}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &delta}},
			})

		case "reasoning-delta":
			if event.Text == "" {
				break
			}
			delta := api.OpenAIDelta{ReasoningContent: event.Text, Reasoning: event.Text}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &delta}},
			})

		case "tool-use":
			toolCalls := []api.OpenAIDeltaToolCall{{
				Index:    toolCallIndex,
				ID:       event.ToolCallID,
				Type:     "function",
				Function: &api.OpenAIDeltaFunction{Name: event.ToolName},
			}}
			delta := api.OpenAIDelta{ToolCalls: toolCalls}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &delta}},
			})
			toolCallIndex++

		case "tool-delta":
			toolCalls := []api.OpenAIDeltaToolCall{{
				Index:    toolCallIndex - 1,
				Function: &api.OpenAIDeltaFunction{Arguments: event.Text},
			}}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &api.OpenAIDelta{ToolCalls: toolCalls}}},
			})

		case "tool-input-start":
			if _, ok := toolCallIndexes[event.ID]; !ok {
				toolCallIndexes[event.ID] = toolCallIndex
				toolCallIndex++
			}
			delta := api.OpenAIDelta{ToolCalls: []api.OpenAIDeltaToolCall{{
				Index: toolCallIndexes[event.ID],
				ID:    event.ID,
				Type:  "function",
				Function: &api.OpenAIDeltaFunction{
					Name: event.ToolName,
				},
			}}}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &delta}},
			})

		case "tool-input-delta":
			idx, ok := toolCallIndexes[event.ID]
			if !ok {
				idx = toolCallIndex
				toolCallIndexes[event.ID] = idx
				toolCallIndex++
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &api.OpenAIDelta{ToolCalls: []api.OpenAIDeltaToolCall{{
					Index:    idx,
					Function: &api.OpenAIDeltaFunction{Arguments: event.Delta},
				}}}}},
			})

		case "tool-call":
			if _, alreadyStreamed := toolCallIndexes[event.ToolCallID]; alreadyStreamed {
				return true
			}
			idx := toolCallIndex
			toolCallIndexes[event.ToolCallID] = idx
			toolCallIndex++
			args := ""
			if event.Input != nil {
				if data, err := json.Marshal(event.Input); err == nil {
					args = string(data)
				}
			}
			delta := api.OpenAIDelta{ToolCalls: []api.OpenAIDeltaToolCall{{
				Index: idx,
				ID:    event.ToolCallID,
				Type:  "function",
				Function: &api.OpenAIDeltaFunction{
					Name:      event.ToolName,
					Arguments: args,
				},
			}}}
			if !sentRole {
				delta.Role = "assistant"
				sentRole = true
			}
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{Index: 0, Delta: &delta}},
			})

		case "finish":
			reason := normalizeFinishReason(event.FinishReason)
			p.WriteSSE(w, flusher, api.OpenAIChatResponse{
				ID:      requestID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []api.OpenAIChoice{{
					Index:        0,
					Delta:        &api.OpenAIDelta{},
					FinishReason: &reason,
				}},
			})
			if event.TotalUsage != nil {
				usage = &api.OpenAIUsage{
					PromptTokens:     event.TotalUsage.InputTokens,
					CompletionTokens: event.TotalUsage.OutputTokens,
					TotalTokens:      event.TotalUsage.InputTokens + event.TotalUsage.OutputTokens,
				}
			}
			if includeUsage && usage != nil {
				p.WriteSSE(w, flusher, api.OpenAIChatResponse{
					ID:      requestID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []api.OpenAIChoice{},
					Usage:   usage,
				})
			}
			p.writeSSEDone(w, flusher)
			finished = true
			return false

		case "error":
			msg := "upstream stream error"
			if event.Error != nil && event.Error.Message != "" {
				msg = event.Error.Message
			}
			log.Printf("[ERROR] Stream error: %s", msg)
			p.writeSSEError(w, flusher, msg, "api_error")
			finished = true
			return false
		}

		return true
	})

	if readErr != nil && !errors.Is(readErr, context.Canceled) {
		log.Printf("[ERROR] Upstream stream read error: %v", readErr)
	}

	if finished || r.Context().Err() != nil {
		return
	}

	// Upstream closed without a terminal "finish" event: synthesize a clean
	// completion so the client is not left waiting for more chunks.
	reason := "stop"
	p.WriteSSE(w, flusher, api.OpenAIChatResponse{
		ID:      requestID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []api.OpenAIChoice{{Index: 0, Delta: &api.OpenAIDelta{}, FinishReason: &reason}},
	})
	if includeUsage && usage != nil {
		p.WriteSSE(w, flusher, api.OpenAIChatResponse{
			ID:      requestID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []api.OpenAIChoice{},
			Usage:   usage,
		})
	}
	p.writeSSEDone(w, flusher)
}

// WriteSSE writes a Server-Sent Event
func (p *Proxy) WriteSSE(w io.Writer, flusher http.Flusher, resp api.OpenAIChatResponse) {
	data, _ := json.Marshal(resp)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// writeSSEDone writes the terminal "[DONE]" SSE sentinel.
func (p *Proxy) writeSSEDone(w io.Writer, flusher http.Flusher) {
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// writeSSEError surfaces an upstream/mid-stream error to a client that is
// already in streaming mode (headers/200 already sent) by emitting an OpenAI
// error payload followed by the "[DONE]" sentinel, so the client stops cleanly
// instead of hanging on a truncated stream.
func (p *Proxy) writeSSEError(w io.Writer, flusher http.Flusher, message, errType string) {
	payload := api.OpenAIErrorResponse{Error: api.OpenAIError{Message: message, Type: errType}}
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", data)
	p.writeSSEDone(w, flusher)
}

// forEachLine reads r line by line and invokes fn for each newline-delimited
// chunk (the trailing newline is included). Unlike bufio.Scanner it has no fixed
// token-size limit, so arbitrarily large lines (e.g. big tool inputs or base64
// payloads) are handled without "token too long" failures. Iteration stops when
// fn returns false or the reader is exhausted.
func forEachLine(r io.Reader, fn func([]byte) bool) error {
	reader := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if !fn(line) {
				return nil
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// parseCCEvent parses one upstream line into a CCStreamEvent. It tolerates both
// raw NDJSON and SSE "data:" framing, and reports ok=false for blank lines, SSE
// comments/heartbeats, and the "[DONE]" sentinel.
func parseCCEvent(raw []byte) (api.CCStreamEvent, bool) {
	line := bytes.TrimSpace(raw)
	if len(line) == 0 || line[0] == ':' {
		return api.CCStreamEvent{}, false
	}
	if rest, ok := bytes.CutPrefix(line, []byte("data:")); ok {
		line = bytes.TrimSpace(rest)
	}
	if len(line) == 0 || bytes.Equal(line, []byte("[DONE]")) {
		return api.CCStreamEvent{}, false
	}
	var event api.CCStreamEvent
	if err := json.Unmarshal(line, &event); err != nil {
		return api.CCStreamEvent{}, false
	}
	return event, true
}

// NonStreamResponse handles non-streaming response
func (p *Proxy) NonStreamResponse(w http.ResponseWriter, ccResp *http.Response, requestID, model string, created int64) {
	var content strings.Builder
	var reasoning strings.Builder
	var inputTokens, outputTokens int
	var hasToolCalls bool
	var toolCalls []api.ToolCall
	var streamErr string
	toolCallByID := map[string]int{}
	toolInputBuffers := map[string]*strings.Builder{}

	readErr := forEachLine(ccResp.Body, func(raw []byte) bool {
		event, ok := parseCCEvent(raw)
		if !ok {
			return true
		}
		p.debugf("[DEBUG] CommandCode stream line: %s", truncateLog(string(bytes.TrimSpace(raw))))

		switch event.Type {
		case "text-delta":
			content.WriteString(event.Text)
		case "reasoning-delta":
			reasoning.WriteString(event.Text)
		case "tool-use":
			hasToolCalls = true
			toolCallByID[event.ToolCallID] = len(toolCalls)
			toolCalls = append(toolCalls, api.ToolCall{
				ID:   event.ToolCallID,
				Type: "function",
				Function: api.FunctionCall{
					Name:      event.ToolName,
					Arguments: "",
				},
			})
		case "tool-delta":
			if len(toolCalls) > 0 {
				toolCalls[len(toolCalls)-1].Function.Arguments += event.Text
			}
		case "tool-input-start":
			hasToolCalls = true
			toolCallByID[event.ID] = len(toolCalls)
			toolInputBuffers[event.ID] = &strings.Builder{}
			toolCalls = append(toolCalls, api.ToolCall{
				ID:   event.ID,
				Type: "function",
				Function: api.FunctionCall{
					Name:      event.ToolName,
					Arguments: "",
				},
			})
		case "tool-input-delta":
			if b := toolInputBuffers[event.ID]; b != nil {
				b.WriteString(event.Delta)
			}
			if idx, ok := toolCallByID[event.ID]; ok {
				toolCalls[idx].Function.Arguments += event.Delta
			}
		case "tool-call":
			hasToolCalls = true
			args := ""
			if event.Input != nil {
				if data, err := json.Marshal(event.Input); err == nil {
					args = string(data)
				}
			}
			if idx, ok := toolCallByID[event.ToolCallID]; ok {
				toolCalls[idx].Function.Name = event.ToolName
				if args != "" {
					toolCalls[idx].Function.Arguments = args
				}
			} else {
				toolCallByID[event.ToolCallID] = len(toolCalls)
				toolCalls = append(toolCalls, api.ToolCall{
					ID:   event.ToolCallID,
					Type: "function",
					Function: api.FunctionCall{
						Name:      event.ToolName,
						Arguments: args,
					},
				})
			}
		case "finish":
			if event.TotalUsage != nil {
				inputTokens = event.TotalUsage.InputTokens
				outputTokens = event.TotalUsage.OutputTokens
			}
		case "error":
			if event.Error != nil && event.Error.Message != "" {
				streamErr = event.Error.Message
			} else {
				streamErr = "upstream stream error"
			}
			log.Printf("[ERROR] Stream error: %s", streamErr)
		}

		return true
	})

	if readErr != nil {
		log.Printf("[ERROR] Upstream stream read error: %v", readErr)
		if streamErr == "" {
			streamErr = "failed to read upstream response"
		}
	}

	// Only fail the request when the upstream produced nothing usable; a partial
	// answer that still carried an error event is returned as-is so the client
	// keeps whatever content/tool calls arrived.
	if streamErr != "" && content.Len() == 0 && reasoning.Len() == 0 && !hasToolCalls {
		p.writeOpenAIError(w, http.StatusBadGateway, streamErr, "api_error")
		return
	}

	msg := &api.OpenAIMessage{
		Role:    "assistant",
		Content: content.String(),
	}
	if reasoning.Len() > 0 {
		msg.ReasoningContent = reasoning.String()
		msg.Reasoning = reasoning.String()
	}
	finishReason := "stop"
	if hasToolCalls {
		msg.Content = nil
		msg.ToolCalls = toolCalls
		finishReason = "tool_calls"
	}

	response := api.OpenAIChatResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []api.OpenAIChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: &finishReason,
		}},
		Usage: &api.OpenAIUsage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (p *Proxy) HandleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		p.writeOpenAIError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		p.writeOpenAIError(w, http.StatusBadRequest, "Failed to read body", "invalid_request_error")
		return
	}

	p.debugf("[DEBUG] Client responses request body: %s", truncateLog(string(body)))

	var responsesReq api.OpenAIResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		p.writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("Invalid JSON: %s", err.Error()), "invalid_request_error")
		return
	}

	chatReq := responsesToChatRequest(responsesReq)
	rewritten, err := json.Marshal(chatReq)
	if err != nil {
		p.writeOpenAIError(w, http.StatusInternalServerError, "Failed to build request", "server_error")
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(rewritten))
	r.ContentLength = int64(len(rewritten))
	p.HandleChatCompletions(w, r)
}

func responsesToChatRequest(req api.OpenAIResponsesRequest) api.OpenAIChatRequest {
	messages := responsesInputToMessages(req.Input)
	if req.Instructions != nil {
		messages = append([]api.OpenAIMessage{{Role: "system", Content: req.Instructions}}, messages...)
	}

	maxTokens := req.MaxCompletionTokens
	if maxTokens == nil {
		maxTokens = req.MaxOutputTokens
	}
	if maxTokens == nil {
		maxTokens = req.MaxTokens
	}

	return api.OpenAIChatRequest{
		Model:               req.Model,
		Messages:            messages,
		Temperature:         req.Temperature,
		MaxTokens:           req.MaxTokens,
		MaxCompletionTokens: maxTokens,
		Stream:              req.Stream,
		Tools:               req.Tools,
		ToolChoice:          req.ToolChoice,
		ParallelToolCalls:   req.ParallelToolCalls,
		ResponseFormat:      req.ResponseFormat,
		Stop:                req.Stop,
		TopP:                req.TopP,
		ReasoningEffort:     responsesReasoningEffort(req),
		User:                req.User,
	}
}

// responsesReasoningEffort extracts the reasoning effort from a Responses API
// request, accepting either a top-level "reasoning_effort" string or the
// "reasoning": {"effort": "..."} object used by the OpenAI Responses API.
func responsesReasoningEffort(req api.OpenAIResponsesRequest) string {
	if req.ReasoningEffort != "" {
		return req.ReasoningEffort
	}
	if r, ok := req.Reasoning.(map[string]any); ok {
		if effort, ok := r["effort"].(string); ok {
			return effort
		}
	}
	return ""
}

func responsesInputToMessages(input any) []api.OpenAIMessage {
	switch v := input.(type) {
	case nil:
		return nil
	case string:
		return []api.OpenAIMessage{{Role: "user", Content: v}}
	case []any:
		if messages := responseItemsToMessages(v); len(messages) > 0 {
			return messages
		}
		return []api.OpenAIMessage{{Role: "user", Content: v}}
	default:
		return []api.OpenAIMessage{{Role: "user", Content: v}}
	}
}

func responseItemsToMessages(items []any) []api.OpenAIMessage {
	messages := make([]api.OpenAIMessage, 0, len(items))
	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "" {
			role = "user"
		}
		content := m["content"]
		if content == nil {
			content = m["text"]
		}
		if content == nil {
			content = m["input"]
		}
		messages = append(messages, api.OpenAIMessage{Role: role, Content: content})
	}
	return messages
}

// HandleModels handles the /v1/models endpoint. The catalog is fetched live from
// the CommandCode provider endpoint and cached for 30 minutes (see internal/models).
func (p *Proxy) HandleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models.List())
}
