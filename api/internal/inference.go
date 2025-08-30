package internal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

type PredictionResponse struct {
	Predictions []struct {
		StationID                  string `json:"station_id"`
		PredictedAvailabilityClass int    `json:"predicted_availability_class"`
		PredictionTime             string `json:"prediction_time"`
		HorizonHours               int    `json:"horizon_hours"`
		AvailabilityPrediction     string `json:"availability_prediction"`
	} `json:"predictions"`
	Count     int    `json:"count"`
	Timestamp string `json:"timestamp"`
}

func (p *PredictionResponse) Validate() error {
	if len(p.Predictions) == 0 {
		return errors.New("no predictions in response")
	}
	if p.Count != len(p.Predictions) {
		return errors.New("prediction count mismatch")
	}
	for i, pred := range p.Predictions {
		if pred.StationID == "" {
			return fmt.Errorf("prediction %d missing station ID", i)
		}
		if pred.PredictionTime == "" {
			return fmt.Errorf("prediction %d missing prediction time", i)
		}
	}
	return nil
}

type MLService struct {
	client  *http.Client
	baseURL string
}

func NewMLService(config *Config) *MLService {
	return &MLService{
		client: &http.Client{
			Timeout: time.Duration(config.ML.RequestTimeoutMin) * time.Minute,
		},
		baseURL: config.ML.ServiceURL,
	}
}

func (m *MLService) GetPredictions(ctx context.Context) (*PredictionResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/predict", nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ML service request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service error %d: %s", resp.StatusCode, string(body))
	}

	var predictionResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predictionResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if err := predictionResp.Validate(); err != nil {
		return nil, fmt.Errorf("invalid response: %w", err)
	}

	log.Printf("ML inference completed: %d predictions generated", predictionResp.Count)
	return &predictionResp, nil
}

func (m *MLService) GetStatus(ctx context.Context) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", m.baseURL+"/status", nil)
	if err != nil {
		return nil, fmt.Errorf("create status request: %w", err)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("status request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status request failed with status %d", resp.StatusCode)
	}

	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, fmt.Errorf("decode status response: %w", err)
	}

	return status, nil
}

type InferenceService struct {
	mlService MLServiceInterface
	database  DatabaseInterface
}

func NewInferenceService(mlService MLServiceInterface, database DatabaseInterface) *InferenceService {
	return &InferenceService{
		mlService: mlService,
		database:  database,
	}
}

func (s *InferenceService) RunInferenceWithResults(ctx context.Context) error {
	resp, err := s.mlService.GetPredictions(ctx)
	if err != nil {
		return fmt.Errorf("get predictions: %w", err)
	}

	predictions, err := s.convertPredictions(resp.Predictions)
	if err != nil {
		return fmt.Errorf("convert predictions: %w", err)
	}

	if err := s.database.InsertPredictions(ctx, predictions); err != nil {
		return fmt.Errorf("store predictions: %w", err)
	}

	return nil
}

func (s *InferenceService) convertPredictions(rawPredictions []struct {
	StationID                  string `json:"station_id"`
	PredictedAvailabilityClass int    `json:"predicted_availability_class"`
	PredictionTime             string `json:"prediction_time"`
	HorizonHours               int    `json:"horizon_hours"`
	AvailabilityPrediction     string `json:"availability_prediction"`
}) ([]Prediction, error) {
	predictions := make([]Prediction, len(rawPredictions))
	
	for i, pred := range rawPredictions {
		predTime, err := time.Parse(time.RFC3339, pred.PredictionTime)
		if err != nil {
			log.Printf("Warning: failed to parse prediction time '%s' for station %s: %v, using current time", 
				pred.PredictionTime, pred.StationID, err)
			predTime = time.Now()
		}

		predictions[i] = Prediction{
			StationID:                  pred.StationID,
			PredictedAvailabilityClass: pred.PredictedAvailabilityClass,
			PredictionTime:             predTime,
			HorizonHours:               pred.HorizonHours,
			AvailabilityPrediction:     pred.AvailabilityPrediction,
		}
	}
	
	return predictions, nil
}
