package internal

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadConfig(t *testing.T) {
	tests := []struct {
		name     string
		envVars  map[string]string
		expected *Config
	}{
		{
			name:    "default config",
			envVars: map[string]string{},
			expected: &Config{
				Database: DatabaseConfig{
					URL: "",
				},
				Server: ServerConfig{
					Port:        "8080",
					Environment: "",
				},
				Divvy: DivvyConfig{
					StationInfoURL:   "https://gbfs.divvybikes.com/gbfs/en/station_information.json",
					StationStatusURL: "https://gbfs.divvybikes.com/gbfs/en/station_status.json",
				},
				ML: MLConfig{
					ServiceURL:        "http://ml:5000",
					RequestTimeoutMin: 5,
					Port:              5000,
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 15,
					PredictionIntervalHours:   2,
					ServerShutdownTimeoutSec:  10,
					MLServiceMaxWaitMin:       5,
					MLServiceCheckIntervalSec: 10,
				},
			},
		},
		{
			name: "custom config with environment variables",
			envVars: map[string]string{
				"DB_URL":                     "postgres://user:pass@db:5432/divvy?sslmode=require",
				"SERVER_PORT":                "9090",
				"ENVIRONMENT":                "production",
				"ML_SERVICE_URL":             "http://ml-service:8000",
				"DATA_COLLECTION_INTERVAL_MIN": "10",
			},
			expected: &Config{
				Database: DatabaseConfig{
					URL: "postgres://user:pass@db:5432/divvy?sslmode=require",
				},
				Server: ServerConfig{
					Port:        "9090",
					Environment: "production",
				},
				Divvy: DivvyConfig{
					StationInfoURL:   "https://gbfs.divvybikes.com/gbfs/en/station_information.json",
					StationStatusURL: "https://gbfs.divvybikes.com/gbfs/en/station_status.json",
				},
				ML: MLConfig{
					ServiceURL:        "http://ml-service:8000",
					RequestTimeoutMin: 5,
					Port:              5000,
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 10,
					PredictionIntervalHours:   2,
					ServerShutdownTimeoutSec:  10,
					MLServiceMaxWaitMin:       5,
					MLServiceCheckIntervalSec: 10,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set env vars
			for k, v := range tt.envVars {
				os.Setenv(k, v)
			}

			// Clean up
			defer func() {
				for k := range tt.envVars {
					os.Unsetenv(k)
				}
			}()

			config := LoadConfig()
			assert.Equal(t, tt.expected, config)
		})
	}
}

func TestConfig_Validate(t *testing.T) {
	tests := []struct {
		name      string
		config    *Config
		expectErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				Database: DatabaseConfig{
					URL: "postgres://user:pass@localhost:5432/db",
				},
				Server: ServerConfig{
					Port: "8080",
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 15,
				},
			},
			expectErr: false,
		},
		{
			name: "missing DB_URL",
			config: &Config{
				Database: DatabaseConfig{
					URL: "",
				},
				Server: ServerConfig{
					Port: "8080",
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 15,
				},
			},
			expectErr: true,
		},
		{
			name: "invalid data collection interval",
			config: &Config{
				Database: DatabaseConfig{
					URL: "postgres://user:pass@localhost:5432/db",
				},
				Server: ServerConfig{
					Port: "8080",
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 0,
				},
			},
			expectErr: true,
		},
		{
			name: "missing server port",
			config: &Config{
				Database: DatabaseConfig{
					URL: "postgres://user:pass@localhost:5432/db",
				},
				Server: ServerConfig{
					Port: "",
				},
				Timing: TimingConfig{
					DataCollectionIntervalMin: 15,
				},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
