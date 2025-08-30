package internal

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestMLService_GetPredictions(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse string
		serverStatus   int
		expectErr      bool
		expectedCount  int
	}{
		{
			name:         "success",
			serverStatus: http.StatusOK,
			serverResponse: `{
				"predictions": [
					{
						"station_id": "123",
						"predicted_availability_class": 1,
						"prediction_time": "2023-01-01T12:00:00Z",
						"horizon_hours": 6,
						"availability_prediction": "green"
					}
				],
				"count": 1,
				"timestamp": "2023-01-01T12:00:00Z"
			}`,
			expectErr:     false,
			expectedCount: 1,
		},
		{
			name:           "server error",
			serverStatus:   http.StatusInternalServerError,
			serverResponse: `{"error": "Internal server error"}`,
			expectErr:      true,
		},
		{
			name:           "invalid json",
			serverStatus:   http.StatusOK,
			serverResponse: `invalid json`,
			expectErr:      true,
		},
		{
			name:         "invalid response - empty predictions",
			serverStatus: http.StatusOK,
			serverResponse: `{
				"predictions": [],
				"count": 0,
				"timestamp": "2023-01-01T12:00:00Z"
			}`,
			expectErr: true,
		},
		{
			name:         "invalid response - count mismatch",
			serverStatus: http.StatusOK,
			serverResponse: `{
				"predictions": [
					{
						"station_id": "123",
						"predicted_availability_class": 1,
						"prediction_time": "2023-01-01T12:00:00Z",
						"horizon_hours": 6,
						"availability_prediction": "green"
					}
				],
				"count": 5,
				"timestamp": "2023-01-01T12:00:00Z"
			}`,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "/predict", r.URL.Path)
				w.WriteHeader(tt.serverStatus)
				w.Write([]byte(tt.serverResponse))
			}))
			defer server.Close()

			config := &Config{
				ML: MLConfig{
					ServiceURL:        server.URL,
					RequestTimeoutMin: 1,
				},
			}

			mlService := NewMLService(config)
			result, err := mlService.GetPredictions(context.Background())

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expectedCount, result.Count)
			}
		})
	}
}

func TestMLService_GetStatus(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse string
		serverStatus   int
		expectErr      bool
	}{
		{
			name:         "success",
			serverStatus: http.StatusOK,
			serverResponse: `{
				"status": "ready",
				"predictor_loaded": true
			}`,
			expectErr: false,
		},
		{
			name:         "service error",
			serverStatus: http.StatusInternalServerError,
			expectErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "GET", r.Method)
				assert.Equal(t, "/status", r.URL.Path)
				w.WriteHeader(tt.serverStatus)
				w.Write([]byte(tt.serverResponse))
			}))
			defer server.Close()

			config := &Config{
				ML: MLConfig{
					ServiceURL:        server.URL,
					RequestTimeoutMin: 1,
				},
			}

			mlService := NewMLService(config)
			result, err := mlService.GetStatus(context.Background())

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestInferenceService_RunInferenceWithResults(t *testing.T) {
	tests := []struct {
		name              string
		mlServiceError    error
		mockInsertError   error
		expectErr         bool
		expectedPredCount int
	}{
		{
			name:              "success",
			mlServiceError:    nil,
			mockInsertError:   nil,
			expectErr:         false,
			expectedPredCount: 1,
		},
		{
			name:           "ml service error",
			mlServiceError: assert.AnError,
			expectErr:      true,
		},
		{
			name:              "database error",
			mlServiceError:    nil,
			mockInsertError:   assert.AnError,
			expectErr:         true,
			expectedPredCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockMLService := new(MockMLService)
			mockDB := new(MockDatabase)

			if tt.mlServiceError != nil {
				mockMLService.On("GetPredictions", mock.Anything).Return((*PredictionResponse)(nil), tt.mlServiceError)
			} else {
				response := &PredictionResponse{
					Predictions: []struct {
						StationID                  string `json:"station_id"`
						PredictedAvailabilityClass int    `json:"predicted_availability_class"`
						PredictionTime             string `json:"prediction_time"`
						HorizonHours               int    `json:"horizon_hours"`
						AvailabilityPrediction     string `json:"availability_prediction"`
					}{
						{
							StationID:                  "123",
							PredictedAvailabilityClass: 1,
							PredictionTime:             "2023-01-01T12:00:00Z",
							HorizonHours:               6,
							AvailabilityPrediction:     "green",
						},
					},
					Count: 1,
				}
				mockMLService.On("GetPredictions", mock.Anything).Return(response, nil)

				if tt.mockInsertError != nil {
					mockDB.On("InsertPredictions", mock.Anything, mock.MatchedBy(func(preds []Prediction) bool {
						return len(preds) == tt.expectedPredCount
					})).Return(tt.mockInsertError)
				} else {
					mockDB.On("InsertPredictions", mock.Anything, mock.MatchedBy(func(preds []Prediction) bool {
						return len(preds) == tt.expectedPredCount
					})).Return(nil)
				}
			}

			inferenceService := NewInferenceService(mockMLService, mockDB)
			err := inferenceService.RunInferenceWithResults(context.Background())

			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			mockMLService.AssertExpectations(t)
			if tt.mlServiceError == nil {
				mockDB.AssertExpectations(t)
			}
		})
	}
}

func TestPredictionResponse_Validate(t *testing.T) {
	tests := []struct {
		name     string
		response *PredictionResponse
		expectErr bool
	}{
		{
			name: "valid response",
			response: &PredictionResponse{
				Predictions: []struct {
					StationID                  string `json:"station_id"`
					PredictedAvailabilityClass int    `json:"predicted_availability_class"`
					PredictionTime             string `json:"prediction_time"`
					HorizonHours               int    `json:"horizon_hours"`
					AvailabilityPrediction     string `json:"availability_prediction"`
				}{
					{
						StationID:      "123",
						PredictionTime: "2023-01-01T12:00:00Z",
					},
				},
				Count: 1,
			},
			expectErr: false,
		},
		{
			name: "empty predictions",
			response: &PredictionResponse{
				Predictions: []struct {
					StationID                  string `json:"station_id"`
					PredictedAvailabilityClass int    `json:"predicted_availability_class"`
					PredictionTime             string `json:"prediction_time"`
					HorizonHours               int    `json:"horizon_hours"`
					AvailabilityPrediction     string `json:"availability_prediction"`
				}{},
				Count: 0,
			},
			expectErr: true,
		},
		{
			name: "count mismatch",
			response: &PredictionResponse{
				Predictions: []struct {
					StationID                  string `json:"station_id"`
					PredictedAvailabilityClass int    `json:"predicted_availability_class"`
					PredictionTime             string `json:"prediction_time"`
					HorizonHours               int    `json:"horizon_hours"`
					AvailabilityPrediction     string `json:"availability_prediction"`
				}{
					{
						StationID:      "123",
						PredictionTime: "2023-01-01T12:00:00Z",
					},
				},
				Count: 5,
			},
			expectErr: true,
		},
		{
			name: "missing station ID",
			response: &PredictionResponse{
				Predictions: []struct {
					StationID                  string `json:"station_id"`
					PredictedAvailabilityClass int    `json:"predicted_availability_class"`
					PredictionTime             string `json:"prediction_time"`
					HorizonHours               int    `json:"horizon_hours"`
					AvailabilityPrediction     string `json:"availability_prediction"`
				}{
					{
						StationID:      "",
						PredictionTime: "2023-01-01T12:00:00Z",
					},
				},
				Count: 1,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.response.Validate()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
