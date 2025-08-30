package internal

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

const (
    queryUpsertStation = `
        INSERT INTO stations (station_id, name, lat, lon, capacity)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (station_id)
        DO UPDATE SET
            name = EXCLUDED.name,
            lat = EXCLUDED.lat,
            lon = EXCLUDED.lon,
            capacity = EXCLUDED.capacity,
            updated_at = CURRENT_TIMESTAMP`

    queryInsertPrediction = `
        INSERT INTO predictions (station_id, predicted_availability_class, availability_prediction, prediction_time, horizon_hours)
        VALUES ($1, $2, $3, $4, $5)`
)

type Database struct {
	db *sql.DB
}

func NewDatabase(cfg *Config) (*Database, error) {
	if cfg.Database.URL == "" {
		return nil, fmt.Errorf("DB_URL is required but not provided")
	}

	db, err := sql.Open("postgres", cfg.Database.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool for cloud database
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	log.Println("Successfully connected to database")
	return &Database{db: db}, nil
}

func (d *Database) Close() error {
	return d.db.Close()
}

func (d *Database) UpsertStations(ctx context.Context, stations []Station) error {
	if len(stations) == 0 {
		return nil
	}

	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, queryUpsertStation)
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, station := range stations {
		_, err := stmt.ExecContext(ctx, station.StationID, station.Name, station.Lat, station.Lon, station.Capacity)
		if err != nil {
			return fmt.Errorf("exec station %s: %w", station.StationID, err)
		}
	}

	return tx.Commit()
}

func (d *Database) InsertAvailabilities(ctx context.Context, availabilities []StationAvailability) error {
	if len(availabilities) == 0 {
		return nil
	}

	query := `
		INSERT INTO station_availability
		(station_id, num_bikes_available, num_docks_available, is_installed, is_renting, is_returning, last_reported)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`

	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, availability := range availabilities {
		_, err := stmt.ExecContext(ctx,
			availability.StationID,
			availability.NumBikesAvailable,
			availability.NumDocksAvailable,
			availability.IsInstalled,
			availability.IsRenting,
			availability.IsReturning,
			availability.LastReported,
		)
		if err != nil {
			return fmt.Errorf("exec availability %s: %w", availability.StationID, err)
		}
	}

	return tx.Commit()
}

func (d *Database) GetStationsWithAvailability(ctx context.Context) ([]StationWithAvailability, error) {
	query := `
		SELECT
			s.station_id, s.name, s.lat, s.lon, s.capacity, s.updated_at,
			COALESCE(sa.num_bikes_available, 0) as num_bikes_available,
			COALESCE(sa.num_docks_available, 0) as num_docks_available,
			COALESCE(sa.is_installed, 0) as is_installed,
			COALESCE(sa.is_renting, 0) as is_renting,
			COALESCE(sa.is_returning, 0) as is_returning,
			COALESCE(sa.last_reported, 0) as last_reported
		FROM stations s
		LEFT JOIN LATERAL (
			SELECT * FROM station_availability
			WHERE station_id = s.station_id
			ORDER BY recorded_at DESC
			LIMIT 1
		) sa ON true
		ORDER BY s.name`

	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var stations []StationWithAvailability
	for rows.Next() {
		var station StationWithAvailability
		err := rows.Scan(
			&station.StationID, &station.Name, &station.Lat, &station.Lon, &station.Capacity, &station.UpdatedAt,
			&station.NumBikesAvailable, &station.NumDocksAvailable,
			&station.IsInstalled, &station.IsRenting, &station.IsReturning, &station.LastReported,
		)
		if err != nil {
			return nil, err
		}
		stations = append(stations, station)
	}

	return stations, nil
}

func (d *Database) GetRecentAvailability(ctx context.Context) ([]StationAvailability, error) {
	query := `
		SELECT id, station_id, num_bikes_available, num_docks_available,
		       is_installed, is_renting, is_returning, last_reported, recorded_at
		FROM station_availability
		WHERE recorded_at > NOW() - INTERVAL '20 minutes'
		ORDER BY recorded_at DESC`

	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []StationAvailability
	for rows.Next() {
		var record StationAvailability
		err := rows.Scan(
			&record.ID, &record.StationID, &record.NumBikesAvailable,
			&record.NumDocksAvailable, &record.IsInstalled, &record.IsRenting,
			&record.IsReturning, &record.LastReported, &record.RecordedAt,
		)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}

	return records, nil
}

func (d *Database) GetAvailabilitySince(ctx context.Context, since time.Time) ([]StationAvailability, error) {
	query := `
		SELECT id, station_id, num_bikes_available, num_docks_available,
		       is_installed, is_renting, is_returning, last_reported, recorded_at
		FROM station_availability
		WHERE recorded_at > $1
		ORDER BY recorded_at ASC`

	rows, err := d.db.QueryContext(ctx, query, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []StationAvailability
	for rows.Next() {
		var record StationAvailability
		err := rows.Scan(
			&record.ID, &record.StationID, &record.NumBikesAvailable,
			&record.NumDocksAvailable, &record.IsInstalled, &record.IsRenting,
			&record.IsReturning, &record.LastReported, &record.RecordedAt,
		)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}

	return records, nil
}

func (d *Database) withTransaction(ctx context.Context, fn func(*sql.Tx) error) error {
    tx, err := d.db.BeginTx(ctx, nil)
    if err != nil {
        return fmt.Errorf("begin transaction: %w", err)
    }
    
    defer func() {
        if err := tx.Rollback(); err != nil && err != sql.ErrTxDone {
            log.Printf("Error rolling back transaction: %v", err)
        }
    }()

    if err := fn(tx); err != nil {
        return err
    }

    return tx.Commit()
}

func (d *Database) InsertPredictions(ctx context.Context, predictions []Prediction) error {
    if len(predictions) == 0 {
        return nil
    }

    return d.withTransaction(ctx, func(tx *sql.Tx) error {
        stmt, err := tx.PrepareContext(ctx, queryInsertPrediction)
        if err != nil {
            return fmt.Errorf("prepare statement: %w", err)
        }
        defer stmt.Close()

        for _, pred := range predictions {
            if _, err := stmt.ExecContext(ctx, pred.StationID, pred.PredictedAvailabilityClass,
                pred.AvailabilityPrediction, pred.PredictionTime, pred.HorizonHours); err != nil {
                return fmt.Errorf("insert prediction for station %s: %w", pred.StationID, err)
            }
        }
        return nil
    })
}

func (d *Database) GetLatestPredictions(ctx context.Context) ([]Prediction, error) {
	query := `
		SELECT DISTINCT ON (station_id)
			id, station_id, predicted_availability_class, availability_prediction,
			prediction_time, horizon_hours, created_at
		FROM predictions
		ORDER BY station_id, created_at DESC`

	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query predictions: %w", err)
	}
	defer rows.Close()

	var predictions []Prediction
	for rows.Next() {
		var p Prediction
		err := rows.Scan(&p.ID, &p.StationID, &p.PredictedAvailabilityClass,
			&p.AvailabilityPrediction, &p.PredictionTime, &p.HorizonHours, &p.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan prediction: %w", err)
		}
		predictions = append(predictions, p)
	}
	return predictions, nil
}

func (d *Database) HealthCheck(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	return d.db.PingContext(ctx)
}

func (d *Database) ExecMigration(ctx context.Context, sql string) error {
	_, err := d.db.ExecContext(ctx, sql)
	return err
}
