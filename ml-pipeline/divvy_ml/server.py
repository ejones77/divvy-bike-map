import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from divvy_ml.pipelines.predictor import DivvyPredictor
from divvy_ml.config import ml_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
predictor_instance = None
last_prediction_time = None
prediction_status = "not_started"
cached_predictions = None
cache_duration_minutes = 15  # Cache predictions for 15 minutes

def initialize_predictor():
    """Initialize the predictor instance during startup"""
    global predictor_instance, prediction_status
    try:
        logger.info("ML_SERVER: initializing_predictor")
        predictor_instance = DivvyPredictor()
        prediction_status = "ready"
        logger.info("ML_SERVER: predictor_initialized_successfully")
    except Exception as e:
        logger.error(f"ML_SERVER: predictor_initialization_failed error={e}")
        prediction_status = "initialization_failed"
        raise

prediction_requests = Counter('ml_prediction_requests_total', 'Total prediction requests')
prediction_duration = Histogram('ml_prediction_duration_seconds', 'Time spent on predictions')

def is_cache_valid():
    """Check if cached predictions are still valid"""
    global last_prediction_time, cached_predictions
    if not cached_predictions or not last_prediction_time:
        return False
    
    cache_expiry = last_prediction_time + timedelta(minutes=cache_duration_minutes)
    return datetime.now(timezone.utc) < cache_expiry

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for container orchestration"""
    global predictor_instance, prediction_status
    
    if prediction_status == "initialization_failed":
        return jsonify({
            "status": "unhealthy",
            "reason": "predictor initialization failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictor_loaded": False
        }), 503
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictor_loaded": predictor_instance is not None,
        "prediction_status": prediction_status
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Detailed status information"""
    global predictor_instance, last_prediction_time, prediction_status
    
    status_info = {
        "prediction_status": prediction_status,
        "last_prediction_time": last_prediction_time.isoformat() if last_prediction_time else None,
        "predictor_loaded": predictor_instance is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return jsonify(status_info), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    """Run inference and return predictions as JSON with caching"""
    global predictor_instance, last_prediction_time, prediction_status, cached_predictions
    
    prediction_requests.inc()
    
    try:
        # Check if we have valid cached predictions
        if is_cache_valid():
            logger.info(f"Returning cached predictions ({len(cached_predictions)} entries)")
            return jsonify({
                "predictions": cached_predictions,
                "count": len(cached_predictions),
                "timestamp": last_prediction_time.isoformat(),
                "cached": True
            }), 200
        
        prediction_status = "running"
        logger.info("Starting new inference via HTTP request")
        
        if predictor_instance is None:
            logger.error("Predictor not initialized - this should not happen")
            return jsonify({"error": "Predictor not initialized"}), 500
        
        predictions_df = predictor_instance.run_inference()
        
        if predictions_df is None:
            prediction_status = "failed"
            return jsonify({"error": "Prediction generation failed"}), 500
        
        # Convert DataFrame to JSON-serializable format
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                "station_id": str(row['station_id']),
                "predicted_availability_class": int(row['predicted_availability_class']),
                "prediction_time": row['prediction_time'].isoformat(),
                "horizon_hours": int(row['horizon_hours']),
                "availability_prediction": str(row['availability_prediction'])
            })
        
        # Cache the new predictions
        cached_predictions = predictions
        last_prediction_time = datetime.now(timezone.utc)
        prediction_status = "completed"
        
        logger.info(f"Inference completed successfully. Generated {len(predictions)} predictions")
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": last_prediction_time.isoformat(),
            "cached": False
        }), 200
        
    except Exception as e:
        prediction_status = "failed"
        logger.error(f"Inference failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting ML inference server on port {ml_config.ml_port}")
    initialize_predictor()
    app.run(host='0.0.0.0', port=ml_config.ml_port, debug=False)
