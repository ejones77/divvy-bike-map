package internal

import (
	"context"
	"time"

	"github.com/stretchr/testify/mock"
)

// Test data fixtures
var (
	TestStation = Station{
		StationID: "test-001",
		Name:      "Test Station",
		Lat:       41.8781,
		Lon:       -87.6298,
		Capacity:  15,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	TestAvailability = StationAvailability{
		ID:                1,
		StationID:         "test-001",
		NumBikesAvailable: 5,
		NumDocksAvailable: 10,
		IsInstalled:       1,
		IsRenting:         1,
		IsReturning:       1,
		LastReported:      time.Now().Unix(),
		RecordedAt:        time.Now(),
	}

	TestStationWithAvailability = StationWithAvailability{
		Station:           TestStation,
		NumBikesAvailable: 5,
		NumDocksAvailable: 10,
		IsInstalled:       1,
		IsRenting:         1,
		IsReturning:       1,
		LastReported:      time.Now().Unix(),
	}
)

// MockDatabase implements database interface for testing
type MockDatabase struct {
	mock.Mock
}

func (m *MockDatabase) UpsertStations(ctx context.Context, stations []Station) error {
	args := m.Called(ctx, stations)
	return args.Error(0)
}

func (m *MockDatabase) GetStationsWithAvailability(ctx context.Context) ([]StationWithAvailability, error) {
	args := m.Called(ctx)
	return args.Get(0).([]StationWithAvailability), args.Error(1)
}

func (m *MockDatabase) InsertAvailabilities(ctx context.Context, availabilities []StationAvailability) error {
	args := m.Called(ctx, availabilities)
	return args.Error(0)
}

func (m *MockDatabase) GetRecentAvailability(ctx context.Context) ([]StationAvailability, error) {
	args := m.Called(ctx)
	return args.Get(0).([]StationAvailability), args.Error(1)
}

func (m *MockDatabase) GetAvailabilitySince(ctx context.Context, since time.Time) ([]StationAvailability, error) {
	args := m.Called(ctx, since)
	return args.Get(0).([]StationAvailability), args.Error(1)
}

func (m *MockDatabase) Close() error {
	args := m.Called()
	return args.Error(0)
}

func (m *MockDatabase) InsertPredictions(ctx context.Context, predictions []Prediction) error {
	args := m.Called(ctx, predictions)
	return args.Error(0)
}

func (m *MockDatabase) GetLatestPredictions(ctx context.Context) ([]Prediction, error) {
	args := m.Called(ctx)
	return args.Get(0).([]Prediction), args.Error(1)
}

func (m *MockDatabase) HealthCheck(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

type MockDivvyClient struct {
	mock.Mock
}

func (m *MockDivvyClient) FetchStationData(ctx context.Context) ([]DivvyStation, []DivvyStationStatus, error) {
	args := m.Called(ctx)
	return args.Get(0).([]DivvyStation), args.Get(1).([]DivvyStationStatus), args.Error(2)
}

type MockMLService struct {
	mock.Mock
}

func (m *MockMLService) GetPredictions(ctx context.Context) (*PredictionResponse, error) {
	args := m.Called(ctx)
	return args.Get(0).(*PredictionResponse), args.Error(1)
}

func (m *MockMLService) GetStatus(ctx context.Context) (map[string]interface{}, error) {
	args := m.Called(ctx)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

type MockStationService struct {
	mock.Mock
}

func (m *MockStationService) RefreshStationData(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

type MockInferenceService struct {
	mock.Mock
}

func (m *MockInferenceService) RunInferenceWithResults(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

// Ensure mocks implement the interfaces
var _ DatabaseInterface = (*MockDatabase)(nil)
var _ DivvyClientInterface = (*MockDivvyClient)(nil)
var _ MLServiceInterface = (*MockMLService)(nil)
var _ StationServiceInterface = (*MockStationService)(nil)
var _ InferenceServiceInterface = (*MockInferenceService)(nil)

// Helper functions
func NewTestConfig() *Config {
	return &Config{
		Database: DatabaseConfig{
			URL: "postgres://test_user:test_pass@localhost:5432/test_db?sslmode=disable",
		},
		Server: ServerConfig{
			Port:        "8080",
			Environment: "test",
		},
		ML: MLConfig{
			ServiceURL:        "http://localhost:5000",
			RequestTimeoutMin: 1,
		},
	}
}
