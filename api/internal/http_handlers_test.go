package internal

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestHTTPHandlers_GetStationsJSON(t *testing.T) {
	tests := []struct {
		name           string
		mockReturn     []StationWithAvailability
		mockError      error
		expectedStatus int
		mode           string
		includePreds   bool
	}{
		{
			name:           "success - current mode",
			mockReturn:     []StationWithAvailability{TestStationWithAvailability},
			mockError:      nil,
			expectedStatus: http.StatusOK,
			mode:           "current",
		},
		{
			name:           "success - predicted mode",
			mockReturn:     []StationWithAvailability{TestStationWithAvailability},
			mockError:      nil,
			expectedStatus: http.StatusOK,
			mode:           "predicted",
			includePreds:   true,
		},
		{
			name:           "database error",
			mockReturn:     nil,
			mockError:      assert.AnError,
			expectedStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDB := new(MockDatabase)
			mockClient := new(MockDivvyClient)
			config := NewTestConfig()

			handlers := NewHTTPHandlers(mockDB, mockClient, config)

			mockDB.On("GetStationsWithAvailability", mock.Anything).
				Return(tt.mockReturn, tt.mockError)

			if tt.includePreds {
				mockDB.On("GetLatestPredictions", mock.Anything).
					Return([]Prediction{{StationID: "test-001"}}, nil)
			}

			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.GET("/stations", handlers.GetStationsJSON)

			url := "/stations"
			if tt.mode != "" {
				url += "?mode=" + tt.mode
			}

			req := httptest.NewRequest("GET", url, nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK {
				var response map[string]interface{}
				err := json.Unmarshal(w.Body.Bytes(), &response)
				assert.NoError(t, err)
				assert.Contains(t, response, "stations")
			}

			mockDB.AssertExpectations(t)
		})
	}
}


func TestHTTPHandlers_RefreshStationData(t *testing.T) {
	tests := []struct {
		name           string
		serviceError   error
		expectedStatus int
	}{
		{
			name:           "success",
			serviceError:   nil,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "service error",
			serviceError:   assert.AnError,
			expectedStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDB := new(MockDatabase)
			mockClient := new(MockDivvyClient)
			mockStationService := new(MockStationService)
			config := NewTestConfig()

			handlers := &HTTPHandlers{
				database:         mockDB,
				divvyClient:      mockClient,
				stationService:   mockStationService,
				mlService:        new(MockMLService),
				inferenceService: new(MockInferenceService),
				config:           config,
			}

			mockStationService.On("RefreshStationData", mock.Anything).Return(tt.serviceError)

			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.POST("/refresh", handlers.RefreshStationData)

			req := httptest.NewRequest("POST", "/refresh", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK {
				var response map[string]interface{}
				err := json.Unmarshal(w.Body.Bytes(), &response)
				assert.NoError(t, err)
				assert.Equal(t, "Station data refreshed successfully", response["message"])
			}

			mockStationService.AssertExpectations(t)
		})
	}
}

func TestHTTPHandlers_TriggerInference(t *testing.T) {
	tests := []struct {
		name           string
		serviceError   error
		expectedStatus int
	}{
		{
			name:           "success",
			serviceError:   nil,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "inference error",
			serviceError:   assert.AnError,
			expectedStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDB := new(MockDatabase)
			mockClient := new(MockDivvyClient)
			mockInferenceService := new(MockInferenceService)
			config := NewTestConfig()

			handlers := &HTTPHandlers{
				database:         mockDB,
				divvyClient:      mockClient,
				stationService:   new(MockStationService),
				mlService:        new(MockMLService),
				inferenceService: mockInferenceService,
				config:           config,
			}

			mockInferenceService.On("RunInferenceWithResults", mock.Anything).Return(tt.serviceError)

			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.POST("/inference", handlers.TriggerInference)

			req := httptest.NewRequest("POST", "/inference", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK {
				var response map[string]interface{}
				err := json.Unmarshal(w.Body.Bytes(), &response)
				assert.NoError(t, err)
				assert.Equal(t, "Inference completed", response["message"])
			}

			mockInferenceService.AssertExpectations(t)
		})
	}
}

func TestHTTPHandlers_HealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		predictions    []Prediction
		dbError        error
		expectedStatus int
		expectedHealth string
	}{
		{
			name: "healthy with predictions",
			predictions: []Prediction{
				{StationID: "123", PredictedAvailabilityClass: 1},
			},
			expectedStatus: http.StatusOK,
			expectedHealth: "healthy",
		},
		{
			name:           "unhealthy no predictions",
			predictions:    []Prediction{},
			expectedStatus: http.StatusServiceUnavailable,
			expectedHealth: "unhealthy",
		},
		{
			name:           "unhealthy db error",
			dbError:        assert.AnError,
			expectedStatus: http.StatusServiceUnavailable,
			expectedHealth: "unhealthy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDB := new(MockDatabase)
			mockClient := new(MockDivvyClient)
			config := NewTestConfig()

			if tt.dbError != nil {
				mockDB.On("GetLatestPredictions", mock.Anything).Return(
					([]Prediction)(nil), tt.dbError)
			} else {
				mockDB.On("GetLatestPredictions", mock.Anything).Return(
					tt.predictions, nil)
			}

			handlers := NewHTTPHandlers(mockDB, mockClient, config)

			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.GET("/health", handlers.HealthCheck)

			req := httptest.NewRequest("GET", "/health", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			var response map[string]interface{}
			err := json.Unmarshal(w.Body.Bytes(), &response)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedHealth, response["status"])
			assert.Equal(t, "divvy-api", response["service"])

			mockDB.AssertExpectations(t)
		})
	}
}
