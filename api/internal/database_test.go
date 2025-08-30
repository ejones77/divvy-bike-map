package internal

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStation_Validate(t *testing.T) {
	tests := []struct {
		name      string
		station   Station
		expectErr bool
	}{
		{
			name:      "valid station",
			station:   TestStation,
			expectErr: false,
		},
		{
			name: "empty station ID",
			station: Station{
				StationID: "",
				Name:      "Test",
				Lat:       41.8781,
				Lon:       -87.6298,
				Capacity:  15,
			},
			expectErr: true,
		},
		{
			name: "empty name",
			station: Station{
				StationID: "test-001",
				Name:      "",
				Lat:       41.8781,
				Lon:       -87.6298,
				Capacity:  15,
			},
			expectErr: true,
		},
		{
			name: "negative capacity",
			station: Station{
				StationID: "test-001",
				Name:      "Test",
				Lat:       41.8781,
				Lon:       -87.6298,
				Capacity:  -5,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.station.Validate()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStationAvailability_Validate(t *testing.T) {
	tests := []struct {
		name         string
		availability StationAvailability
		expectErr    bool
	}{
		{
			name:         "valid availability",
			availability: TestAvailability,
			expectErr:    false,
		},
		{
			name: "empty station ID",
			availability: StationAvailability{
				StationID:         "",
				NumBikesAvailable: 5,
				NumDocksAvailable: 10,
			},
			expectErr: true,
		},
		{
			name: "negative bikes available",
			availability: StationAvailability{
				StationID:         "test-001",
				NumBikesAvailable: -1,
				NumDocksAvailable: 10,
			},
			expectErr: true,
		},
		{
			name: "negative docks available",
			availability: StationAvailability{
				StationID:         "test-001",
				NumBikesAvailable: 5,
				NumDocksAvailable: -1,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.availability.Validate()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
