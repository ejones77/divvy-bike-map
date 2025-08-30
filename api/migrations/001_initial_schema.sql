CREATE TABLE IF NOT EXISTS stations (
    station_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    lat DECIMAL(10, 8) NOT NULL,
    lon DECIMAL(11, 8) NOT NULL,
    capacity INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS station_availability (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL REFERENCES stations(station_id),
    num_bikes_available INTEGER NOT NULL,
    num_docks_available INTEGER NOT NULL,
    is_installed SMALLINT NOT NULL DEFAULT 1,
    is_renting SMALLINT NOT NULL DEFAULT 1,
    is_returning SMALLINT NOT NULL DEFAULT 1,
    last_reported BIGINT NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_station_availability_station_id ON station_availability(station_id);
CREATE INDEX IF NOT EXISTS idx_station_availability_recorded_at ON station_availability(recorded_at);
CREATE INDEX IF NOT EXISTS idx_station_availability_last_reported ON station_availability(last_reported);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_stations_updated_at ON stations;
CREATE TRIGGER update_stations_updated_at 
    BEFORE UPDATE ON stations 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_updated_at_column(); 