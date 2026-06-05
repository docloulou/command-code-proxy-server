// Package models exposes the catalog of models offered by CommandCode. The list
// is fetched live from the provider endpoint and cached for 30 minutes so the
// proxy always advertises the current catalog without hammering the upstream.
package models

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/dev2k6/command-code-proxy-server/internal/api"
)

const modelsURL = "https://api.commandcode.ai/provider/v1/models"
const cacheDuration = 30 * time.Minute

type cachedList struct {
	list        api.OpenAIModelList
	lastUpdated time.Time
}

var (
	mu     sync.RWMutex
	cache  *cachedList
	client = &http.Client{Timeout: 10 * time.Second}
)

// List returns the available models. It serves a cached copy when the cache is
// fresh (younger than 30 minutes). When the cache is stale it refetches; on a
// fetch error it falls back to the last good cache, or to a built-in static list
// if nothing has ever been cached.
func List() api.OpenAIModelList {
	mu.RLock()
	c := cache
	mu.RUnlock()

	if c != nil && time.Since(c.lastUpdated) < cacheDuration {
		return c.list
	}

	fresh, err := fetch()
	if err != nil {
		mu.RLock()
		defer mu.RUnlock()
		if cache != nil {
			return cache.list
		}
		return fallbackList()
	}

	mu.Lock()
	cache = &cachedList{list: fresh, lastUpdated: time.Now()}
	mu.Unlock()
	return fresh
}

// fetch retrieves and parses the upstream model catalog.
func fetch() (api.OpenAIModelList, error) {
	resp, err := client.Get(modelsURL)
	if err != nil {
		return api.OpenAIModelList{}, fmt.Errorf("failed to fetch models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return api.OpenAIModelList{}, fmt.Errorf("models endpoint returned %d: %s", resp.StatusCode, string(body))
	}

	var list api.OpenAIModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return api.OpenAIModelList{}, fmt.Errorf("failed to parse models: %w", err)
	}
	if len(list.Data) == 0 {
		return api.OpenAIModelList{}, fmt.Errorf("models endpoint returned an empty list")
	}
	if list.Object == "" {
		list.Object = "list"
	}
	return list, nil
}

// fallbackList is the offline catalog used only when the upstream has never been
// reachable since startup.
func fallbackList() api.OpenAIModelList {
	return api.OpenAIModelList{
		Object: "list",
		Data: []api.OpenAIModel{
			{ID: "moonshotai/Kimi-K2.6", Object: "model", OwnedBy: "command-code"},
			{ID: "moonshotai/Kimi-K2.5", Object: "model", OwnedBy: "command-code"},
			{ID: "zai-org/GLM-5.1", Object: "model", OwnedBy: "command-code"},
			{ID: "zai-org/GLM-5", Object: "model", OwnedBy: "command-code"},
			{ID: "MiniMaxAI/MiniMax-M3", Object: "model", OwnedBy: "command-code"},
			{ID: "MiniMaxAI/MiniMax-M2.7", Object: "model", OwnedBy: "command-code"},
			{ID: "MiniMaxAI/MiniMax-M2.5", Object: "model", OwnedBy: "command-code"},
			{ID: "deepseek/deepseek-v4-pro", Object: "model", OwnedBy: "command-code"},
			{ID: "deepseek/deepseek-v4-flash", Object: "model", OwnedBy: "command-code"},
			{ID: "Qwen/Qwen3.6-Max-Preview", Object: "model", OwnedBy: "command-code"},
			{ID: "Qwen/Qwen3.6-Plus", Object: "model", OwnedBy: "command-code"},
			{ID: "Qwen/Qwen3.7-Max", Object: "model", OwnedBy: "command-code"},
			{ID: "stepfun/Step-3.7-Flash", Object: "model", OwnedBy: "command-code"},
			{ID: "stepfun/Step-3.5-Flash", Object: "model", OwnedBy: "command-code"},
			{ID: "xiaomi/mimo-v2.5-pro", Object: "model", OwnedBy: "command-code"},
			{ID: "xiaomi/mimo-v2.5", Object: "model", OwnedBy: "command-code"},
			{ID: "google/gemini-3.1-flash-lite", Object: "model", OwnedBy: "command-code"},
		},
	}
}

// init warms the cache in the background so the first /v1/models request is fast.
func init() {
	go func() { _ = List() }()
}
