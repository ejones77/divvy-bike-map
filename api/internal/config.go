package internal

import (
	"errors"
	"log"
	"os"
	"strconv"
)

type Config struct {
	Database DatabaseConfig
	Server   ServerConfig
	Divvy    DivvyConfig
	ML       MLConfig
	Timing   TimingConfig
}

type DatabaseConfig struct {
	URL string
}

type ServerConfig struct {
	Port        string
	Environment string
}

type DivvyConfig struct {
	StationInfoURL   string
	StationStatusURL string
}

type MLConfig struct {
	ServiceURL        string
	RequestTimeoutMin int
	Port              int
}

type TimingConfig struct {
	DataCollectionIntervalMin int
	PredictionIntervalHours   int
	ServerShutdownTimeoutSec  int
	MLServiceMaxWaitMin       int
	MLServiceCheckIntervalSec int
}

func LoadConfig() *Config {
	return &Config{
		Database: DatabaseConfig{
			URL: getEnv("DB_URL", ""),
		},
		Server: ServerConfig{
			Port:        getEnv("SERVER_PORT", "8080"),
			Environment: getEnv("ENVIRONMENT", ""),
		},
		Divvy: DivvyConfig{
			StationInfoURL:   getEnv("DIVVY_STATION_INFO_URL", "https://gbfs.divvybikes.com/gbfs/en/station_information.json"),
			StationStatusURL: getEnv("DIVVY_STATION_STATUS_URL", "https://gbfs.divvybikes.com/gbfs/en/station_status.json"),
		},

		ML: MLConfig{
			ServiceURL:        getEnv("ML_SERVICE_URL", "http://ml:5000"),
			RequestTimeoutMin: getEnvInt("ML_REQUEST_TIMEOUT_MIN", 5),
			Port:              getEnvInt("ML_PORT", 5000),
		},

		Timing: TimingConfig{
			DataCollectionIntervalMin: getEnvInt("DATA_COLLECTION_INTERVAL_MIN", 15),
			PredictionIntervalHours:   getEnvInt("PREDICTION_INTERVAL_HOURS", 2),
			ServerShutdownTimeoutSec:  getEnvInt("SERVER_SHUTDOWN_TIMEOUT_SEC", 10),
			MLServiceMaxWaitMin:       getEnvInt("ML_SERVICE_MAX_WAIT_MIN", 5),
			MLServiceCheckIntervalSec: getEnvInt("ML_SERVICE_CHECK_INTERVAL_SEC", 10),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func (c *Config) Validate() error {
	if c.Database.URL == "" {
		return errors.New("DB_URL is required but not provided")
	}
	if c.Timing.DataCollectionIntervalMin <= 0 {
		return errors.New("data collection interval must be positive")
	}
	if c.Server.Port == "" {
		return errors.New("server port is required")
	}
	return nil
}

func getEnvInt(key string, defaultValue int) int {
	val := os.Getenv(key)
	if val == "" {
		return defaultValue
	}
	if intVal, err := strconv.Atoi(val); err == nil {
		return intVal
	}
	log.Printf("Warning: invalid integer value for %s: %s, using default %d", key, val, defaultValue)
	return defaultValue
}
