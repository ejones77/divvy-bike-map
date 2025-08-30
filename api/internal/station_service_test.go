package internal

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestStationService_RefreshStationData(t *testing.T) {
	tests := []struct {
		name               string
		mockStations       []DivvyStation
		mockStatuses       []DivvyStationStatus
		fetchError         error
		upsertError        error
		insertError        error
		expectErr          bool
		expectedUpsertCall int
		expectedInsertCall int
	}{
		{
			name: "success",
			mockStations: []DivvyStation{
				{
					StationID: "123",
					Name:      "Test Station",
					Lat:       41.8781,
					Lon:       -87.6298,
					Capacity:  15,
				},
			},
			mockStatuses: []DivvyStationStatus{
				{
					StationID:         "123",
					NumBikesAvailable: 5,
					NumDocksAvailable: 10,
					IsInstalled:       1,
					IsRenting:         1,
					IsReturning:       1,
					LastReported:      1640995200,
				},
			},
			expectErr:          false,
			expectedUpsertCall: 1,
			expectedInsertCall: 1,
		},
		{
			name:       "fetch error",
			fetchError: assert.AnError,
			expectErr:  true,
		},
		{
			name: "empty data",
			mockStations:       []DivvyStation{},
			mockStatuses:       []DivvyStationStatus{},
			expectErr:          false,
			expectedUpsertCall: 1,
			expectedInsertCall: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDB := new(MockDatabase)
			mockClient := new(MockDivvyClient)

			if tt.fetchError != nil {
				mockClient.On("FetchStationData", mock.Anything).Return(
					([]DivvyStation)(nil), ([]DivvyStationStatus)(nil), tt.fetchError)
			} else {
				mockClient.On("FetchStationData", mock.Anything).Return(
					tt.mockStations, tt.mockStatuses, nil)

				if tt.expectedUpsertCall > 0 {
					mockDB.On("UpsertStations", mock.Anything, mock.MatchedBy(func(stations []Station) bool {
						return len(stations) == len(tt.mockStations)
					})).Return(tt.upsertError).Times(1)
				}

				if tt.expectedInsertCall > 0 {
					mockDB.On("InsertAvailabilities", mock.Anything, mock.MatchedBy(func(availabilities []StationAvailability) bool {
						return len(availabilities) == len(tt.mockStatuses)
					})).Return(tt.insertError).Times(1)
				}
			}

			service := NewStationService(mockDB, mockClient)
			err := service.RefreshStationData(context.Background())

			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			mockClient.AssertExpectations(t)
			if tt.fetchError == nil {
				mockDB.AssertExpectations(t)
			}
		})
	}
}

func TestStationService_ConvertToStation(t *testing.T) {
	service := &StationService{}
	
	divvyStation := DivvyStation{
		StationID: "test-123",
		Name:      "Test Station",
		Lat:       41.8781,
		Lon:       -87.6298,
		Capacity:  20,
	}

	result := service.convertToStation(divvyStation)

	assert.Equal(t, divvyStation.StationID, result.StationID)
	assert.Equal(t, divvyStation.Name, result.Name)
	assert.Equal(t, divvyStation.Lat, result.Lat)
	assert.Equal(t, divvyStation.Lon, result.Lon)
	assert.Equal(t, divvyStation.Capacity, result.Capacity)
}

func TestStationService_ConvertToAvailability(t *testing.T) {
	service := &StationService{}
	
	divvyStatus := DivvyStationStatus{
		StationID:         "test-123",
		NumBikesAvailable: 8,
		NumDocksAvailable: 12,
		IsInstalled:       1,
		IsRenting:         1,
		IsReturning:       1,
		LastReported:      1640995200,
	}

	result := service.convertToAvailability(divvyStatus)

	assert.Equal(t, divvyStatus.StationID, result.StationID)
	assert.Equal(t, divvyStatus.NumBikesAvailable, result.NumBikesAvailable)
	assert.Equal(t, divvyStatus.NumDocksAvailable, result.NumDocksAvailable)
	assert.Equal(t, divvyStatus.IsInstalled, result.IsInstalled)
	assert.Equal(t, divvyStatus.IsRenting, result.IsRenting)
	assert.Equal(t, divvyStatus.IsReturning, result.IsReturning)
	assert.Equal(t, divvyStatus.LastReported, result.LastReported)
}
