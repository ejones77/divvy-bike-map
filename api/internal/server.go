package internal

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Server struct {
	router   *gin.Engine
	handlers *HTTPHandlers
	config   *Config
}

func NewServer(config *Config, handlers *HTTPHandlers) (*Server, error) {

	if config.Server.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.Default()

	return &Server{
		router:   router,
		handlers: handlers,
		config:   config,
	}, nil
}

func (s *Server) setupRoutes() {
	s.router.Static("/static", "./static")

	s.router.LoadHTMLGlob("templates/*")

	s.router.GET("/health", s.handlers.HealthCheck)
	s.router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	s.router.GET("/", s.handlers.HomePage)
	s.router.GET("/stations", s.handlers.GetStationsHTML)
	s.router.GET("/predictions", func(c *gin.Context) {
		c.Request.URL.Path = "/stations"
		c.Request.URL.RawQuery = "mode=predicted"
		s.router.HandleContext(c)
	})

	api := s.router.Group("/api")
	{
		api.GET("/stations", s.handlers.GetStationsHTML)
		api.GET("/stations/json", s.handlers.GetStationsJSON)
		api.POST("/refresh", s.handlers.RefreshStationData)
	}
}

func (s *Server) setupMiddleware() {
	s.router.Use(gin.Logger())
	s.router.Use(gin.Recovery())

	s.router.Use(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")

		// DEBUG: Log all requests
		log.Printf("DEBUG: Request to %s %s from origin: '%s'", c.Request.Method, c.Request.URL.Path, origin)

		// TEMPORARY: Allow everything for debugging
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "*")
		c.Header("Access-Control-Allow-Credentials", "false")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})
}

func (s *Server) Start() error {
	s.setupMiddleware()
	s.setupRoutes()

	s.startDataCollection(context.Background())

	s.StartPredictionService(context.Background())

	server := &http.Server{
		Addr:    ":" + s.config.Server.Port,
		Handler: s.router,
	}

	go func() {
		log.Printf("Server starting on port %s", s.config.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s.config.Timing.ServerShutdownTimeoutSec)*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		return fmt.Errorf("server forced to shutdown: %w", err)
	}

	log.Println("Server exited")
	return nil
}

func (s *Server) startDataCollection(ctx context.Context) {
	go func() {
		now := time.Now()
		interval := time.Duration(s.config.Timing.DataCollectionIntervalMin) * time.Minute
		nextInterval := now.Truncate(interval).Add(interval)
		timeUntilNext := nextInterval.Sub(now)

		log.Printf("Data collection service starting - next fetch at %s (in %v)",
			nextInterval.Format("15:04:05"), timeUntilNext)

		// Wait until the next 15-minute boundary
		select {
		case <-ctx.Done():
			log.Println("Data collection service shutting down before first fetch")
			return
		case <-time.After(timeUntilNext):
			// First fetch at the boundary
			if err := s.handlers.RefreshStationDataInternal(context.Background()); err != nil {
				log.Printf("Initial scheduled data collection failed: %v", err)
			} else {
				log.Printf("Initial scheduled data collection completed at %s", time.Now().Format("15:04:05"))
			}
		}

		// Now start regular 15-minute ticker
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		log.Printf("Data collection service running - fetching every %d minutes on the boundary", s.config.Timing.DataCollectionIntervalMin)

		for {
			select {
			case <-ctx.Done():
				log.Println("Data collection service shutting down")
				return
			case <-ticker.C:
				if err := s.handlers.RefreshStationDataInternal(context.Background()); err != nil {
					log.Printf("Scheduled data collection failed: %v", err)
				} else {
					log.Printf("Scheduled data collection completed at %s", time.Now().Format("15:04:05"))
				}
			}
		}
	}()
}

func (s *Server) waitAndGenerateInitialPredictions(ctx context.Context) error {
	maxWait := time.Duration(s.config.Timing.MLServiceMaxWaitMin) * time.Minute
	checkInterval := time.Duration(s.config.Timing.MLServiceCheckIntervalSec) * time.Second

	start := time.Now()
	for {
		if time.Since(start) > maxWait {
			return fmt.Errorf("timeout waiting for ML service after %v", maxWait)
		}

		// Try to call the ML service directly
		if err := s.handlers.inferenceService.RunInferenceWithResults(ctx); err != nil {
			log.Printf("ML service not ready yet (elapsed: %v): %v", time.Since(start), err)

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(checkInterval):
				continue
			}
		}

		log.Printf("Initial predictions generated successfully after %v", time.Since(start))
		return nil
	}
}

func (s *Server) StartPredictionService(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(s.config.Timing.PredictionIntervalHours) * time.Hour)
	defer ticker.Stop()

	go func() {
		log.Println("Waiting for ML service and generating initial predictions...")
		if err := s.waitAndGenerateInitialPredictions(ctx); err != nil {
			log.Printf("Initial prediction generation failed: %v", err)
		} else {
			log.Printf("Initial predictions generated successfully at %s", time.Now().Format("15:04:05"))
		}

		log.Printf("Prediction service running - generating predictions every %d hours", s.config.Timing.PredictionIntervalHours)

		for {
			select {
			case <-ctx.Done():
				log.Println("Prediction service shutting down")
				return
			case <-ticker.C:
				if err := s.handlers.inferenceService.RunInferenceWithResults(context.Background()); err != nil {
					log.Printf("Scheduled prediction generation failed: %v", err)
				} else {
					log.Printf("Scheduled predictions generated at %s", time.Now().Format("15:04:05"))
				}
			}
		}
	}()
}
