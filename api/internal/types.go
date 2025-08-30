package internal

import (
	"context"
	"errors"
	"time"
)

type Station struct {
	StationID string    `json:"station_id" db:"station_id" validate:"required"`
	Name      string    `json:"name" db:"name" validate:"required"`
	Lat       float64   `json:"lat" db:"lat" validate:"required"`
	Lon       float64   `json:"lon" db:"lon" validate:"required"`
	Capacity  int       `json:"capacity" db:"capacity" validate:"min=0"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

func (s *Station) Validate() error {
	if s.StationID == "" {
		return errors.New("station ID is required")
	}
	if s.Name == "" {
		return errors.New("station name is required")
	}
	if s.Capacity < 0 {
		return errors.New("capacity cannot be negative")
	}
	return nil
}

type StationAvailability struct {
	ID                int       `json:"id" db:"id"`
	StationID         string    `json:"station_id" db:"station_id" validate:"required"`
	NumBikesAvailable int       `json:"num_bikes_available" db:"num_bikes_available" validate:"min=0"`
	NumDocksAvailable int       `json:"num_docks_available" db:"num_docks_available" validate:"min=0"`
	IsInstalled       int       `json:"is_installed" db:"is_installed"`
	IsRenting         int       `json:"is_renting" db:"is_renting"`
	IsReturning       int       `json:"is_returning" db:"is_returning"`
	LastReported      int64     `json:"last_reported" db:"last_reported"`
	RecordedAt        time.Time `json:"recorded_at" db:"recorded_at"`
}

func (sa *StationAvailability) Validate() error {
	if sa.StationID == "" {
		return errors.New("station ID is required")
	}
	if sa.NumBikesAvailable < 0 || sa.NumDocksAvailable < 0 {
		return errors.New("availability counts cannot be negative")
	}
	return nil
}

type DivvyStationInfoResponse struct {
	Data struct {
		Stations []DivvyStation `json:"stations"`
	} `json:"data"`
}

type DivvyStationStatusResponse struct {
	Data struct {
		Stations []DivvyStationStatus `json:"stations"`
	} `json:"data"`
}

type DivvyStation struct {
	StationID string  `json:"station_id"`
	Name      string  `json:"name"`
	Lat       float64 `json:"lat"`
	Lon       float64 `json:"lon"`
	Capacity  int     `json:"capacity"`
}

type DivvyStationStatus struct {
	StationID         string `json:"station_id"`
	NumBikesAvailable int    `json:"num_bikes_available"`
	NumDocksAvailable int    `json:"num_docks_available"`
	IsInstalled       int    `json:"is_installed"`
	IsRenting         int    `json:"is_renting"`
	IsReturning       int    `json:"is_returning"`
	LastReported      int64  `json:"last_reported"`
}

type StationWithAvailability struct {
	Station
	NumBikesAvailable int   `json:"num_bikes_available"`
	NumDocksAvailable int   `json:"num_docks_available"`
	IsInstalled       int   `json:"is_installed"`
	IsRenting         int   `json:"is_renting"`
	IsReturning       int   `json:"is_returning"`
	LastReported      int64 `json:"last_reported"`
}

type Prediction struct {
	ID                         int       `json:"id" db:"id"`
	StationID                  string    `json:"station_id" db:"station_id"`
	PredictedAvailabilityClass int       `json:"predicted_availability_class" db:"predicted_availability_class"`
	AvailabilityPrediction     string    `json:"availability_prediction" db:"availability_prediction"`
	PredictionTime             time.Time `json:"prediction_time" db:"prediction_time"`
	HorizonHours               int       `json:"horizon_hours" db:"horizon_hours"`
	CreatedAt                  time.Time `json:"created_at" db:"created_at"`
}

// Focused repository interfaces following Interface Segregation Principle
type StationRepository interface {
	UpsertStations(ctx context.Context, stations []Station) error
	GetStationsWithAvailability(ctx context.Context) ([]StationWithAvailability, error)
}

type AvailabilityRepository interface {
	InsertAvailabilities(ctx context.Context, availabilities []StationAvailability) error
	GetRecentAvailability(ctx context.Context) ([]StationAvailability, error)
	GetAvailabilitySince(ctx context.Context, since time.Time) ([]StationAvailability, error)
}

type PredictionRepository interface {
	InsertPredictions(ctx context.Context, predictions []Prediction) error
	GetLatestPredictions(ctx context.Context) ([]Prediction, error)
}

type HealthChecker interface {
	HealthCheck(ctx context.Context) error
	Close() error
}

// Combined interface for backward compatibility where needed
type DatabaseInterface interface {
	StationRepository
	AvailabilityRepository
	PredictionRepository
	HealthChecker
}

// Service interfaces
type DivvyClientInterface interface {
	FetchStationData(ctx context.Context) ([]DivvyStation, []DivvyStationStatus, error)
}

type MLServiceInterface interface {
	GetPredictions(ctx context.Context) (*PredictionResponse, error)
	GetStatus(ctx context.Context) (map[string]interface{}, error)
}

type StationServiceInterface interface {
	RefreshStationData(ctx context.Context) error
}

type InferenceServiceInterface interface {
	RunInferenceWithResults(ctx context.Context) error
}
