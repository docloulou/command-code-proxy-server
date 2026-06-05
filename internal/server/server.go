package server

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/dev2k6/command-code-proxy-server/internal/proxy"
)

const defaultPort = "55990"
const defaultHost = "127.0.0.1"

// Server represents the HTTP server
type Server struct {
	Port    string
	Host    string
	Proxy   *proxy.Proxy
	Handler http.Handler
}

// NewServer creates a new server instance
func NewServer(proxy *proxy.Proxy) *Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", logger(proxy.HandleChatCompletions))
	mux.HandleFunc("/chat/completions", logger(proxy.HandleChatCompletions))
	mux.HandleFunc("/v1/responses", logger(proxy.HandleResponses))
	mux.HandleFunc("/v1/models", logger(proxy.HandleModels))
	mux.HandleFunc("/health", logger(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	}))

	return &Server{
		Port:    defaultPort,
		Host:    defaultHost,
		Proxy:   proxy,
		Handler: mux,
	}
}

// SetPort sets the port for the server
func (s *Server) SetPort(port string) {
	if port != "" {
		s.Port = port
	}
}

// SetHost sets the host for the server
func (s *Server) SetHost(host string) {
	if host != "" {
		s.Host = host
	}
}

// GetPort returns the server port
func (s *Server) GetPort() string {
	return s.Port
}

// GetHost returns the server host
func (s *Server) GetHost() string {
	return s.Host
}

// logger is a middleware for logging requests
func logger(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("[%s] %s %s", r.Method, r.URL.Path, r.RemoteAddr)
		next(w, r)
		log.Printf("[%s] %s done in %v", r.Method, r.URL.Path, time.Since(start))
	}
}

// Start starts the HTTP server and blocks until it receives an interrupt or
// termination signal, then shuts down gracefully.
//
// Note: WriteTimeout is intentionally left unset. Streaming (SSE) responses can
// stay open far longer than any fixed write deadline; a non-zero WriteTimeout
// would abruptly cut long completions. ReadHeaderTimeout still guards against
// slow-header (Slowloris) clients and IdleTimeout reaps idle keep-alive conns.
func (s *Server) Start() {
	addr := s.Host + ":" + s.Port
	httpServer := &http.Server{
		Addr:              addr,
		Handler:           s.Handler,
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serverErr := make(chan error, 1)
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		log.Fatalf("Server failed: %v", err)
	case <-ctx.Done():
		stop()
		log.Println("Shutting down server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			log.Printf("Graceful shutdown failed: %v", err)
		}
	}
}
