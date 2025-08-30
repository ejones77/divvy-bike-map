CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL,
    predicted_availability_class INTEGER NOT NULL,
    availability_prediction VARCHAR(10) NOT NULL,
    prediction_time TIMESTAMP WITH TIME ZONE NOT NULL,
    horizon_hours INTEGER NOT NULL DEFAULT 6,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_station_created 
ON predictions(station_id, created_at DESC);
