package internal

import (
	"context"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

type HTTPHandlers struct {
	database          DatabaseInterface
	divvyClient       DivvyClientInterface
	stationService    StationServiceInterface
	mlService         MLServiceInterface
	inferenceService  InferenceServiceInterface
	config            *Config
}

func NewHTTPHandlers(database DatabaseInterface, divvyClient DivvyClientInterface, config *Config) *HTTPHandlers {
	mlService := NewMLService(config)
	inferenceService := NewInferenceService(mlService, database)
	return &HTTPHandlers{
		database:         database,
		divvyClient:      divvyClient,
		stationService:   NewStationService(database, divvyClient),
		mlService:        mlService,
		inferenceService: inferenceService,
		config:           config,
	}
}

func (h *HTTPHandlers) handleError(c *gin.Context, statusCode int, message string, err error) {
	log.Printf("Error in %s %s: %v", c.Request.Method, c.Request.URL.Path, err)
	c.JSON(statusCode, gin.H{"error": message})
}

func (h *HTTPHandlers) HomePage(c *gin.Context) {
	c.HTML(http.StatusOK, "index.html", gin.H{
		"title": "Divvy Bike Availability",
	})
}

func (h *HTTPHandlers) GetStationsHTML(c *gin.Context) {
	ctx := c.Request.Context()
	mode := c.DefaultQuery("mode", "current")

	stations, err := h.database.GetStationsWithAvailability(ctx)
	if err != nil {
		h.handleError(c, http.StatusInternalServerError, "Failed to fetch station data", err)
		return
	}

	predictionsMap := map[string]Prediction{}
	if mode == "predicted" {
		if predictions, err := h.database.GetLatestPredictions(ctx); err == nil && len(predictions) > 0 {
			for _, p := range predictions {
				predictionsMap[p.StationID] = p
			}
		}
	}

	c.HTML(http.StatusOK, "stations.html", gin.H{
		"stations":       stations,
		"predictionsMap": predictionsMap,
		"mode":           mode,
	})
}

func (h *HTTPHandlers) GetStationsJSON(c *gin.Context) {
	ctx := c.Request.Context()
	mode := c.DefaultQuery("mode", "current")

	stations, err := h.database.GetStationsWithAvailability(ctx)
	if err != nil {
		h.handleError(c, http.StatusInternalServerError, "Failed to fetch station data", err)
		return
	}

	response := gin.H{"stations": stations}

	if mode == "predicted" {
		predictions, err := h.database.GetLatestPredictions(ctx)
		if err != nil || len(predictions) == 0 {
			log.Printf("No predictions available: %v", err)
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Predictions not ready"})
			return
		}
		response["predictions"] = predictions
	}

	c.JSON(http.StatusOK, response)
}

func (h *HTTPHandlers) RefreshStationData(c *gin.Context) {
	ctx := c.Request.Context()

	if err := h.stationService.RefreshStationData(ctx); err != nil {
		h.handleError(c, http.StatusInternalServerError, "Failed to refresh station data", err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Station data refreshed successfully"})
}

func (h *HTTPHandlers) RefreshStationDataInternal(ctx context.Context) error {
	return h.stationService.RefreshStationData(ctx)
}

func (h *HTTPHandlers) HealthCheck(c *gin.Context) {
	ctx := c.Request.Context()
	
	predictions, err := h.database.GetLatestPredictions(ctx)
	if err != nil || len(predictions) == 0 {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":  "unhealthy",
			"service": "divvy-api",
			"reason":  "predictions not available",
		})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"status":            "healthy",
		"service":           "divvy-api",
		"predictions_count": len(predictions),
	})
}



func (h *HTTPHandlers) TriggerInference(c *gin.Context) {
	ctx := c.Request.Context()

	err := h.inferenceService.RunInferenceWithResults(ctx)
	if err != nil {
		h.handleError(c, http.StatusInternalServerError, "Inference failed", err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Inference completed"})
}
