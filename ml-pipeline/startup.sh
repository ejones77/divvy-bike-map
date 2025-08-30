#!/bin/bash
set -e

echo "ML_STARTUP: initialization_start" >&2

mkdir -p /app/output

# Check if any XGBoost model directory exists locally
MODEL_EXISTS=false
for dir in /app/xgb_model*; do
    if [ -d "$dir" ]; then
        echo "ML_STARTUP: found_existing_model dir=$dir" >&2
        MODEL_EXISTS=true
        break
    fi
done

if [ "$MODEL_EXISTS" = false ]; then
    echo "ML_STARTUP: no_local_model_found attempting_s3_download" >&2
    
    # Try to download model from S3 using Python
    if uv run python -c "
from divvy_ml.utils.model_loader import download_latest_model_from_s3
import sys
result = download_latest_model_from_s3('/app')
if result:
    print(f'SUCCESS: {result}')
    sys.exit(0)
else:
    print('FAILED: Could not download model from S3')
    sys.exit(1)
"; then
        echo "ML_STARTUP: s3_model_download_success" >&2
    else
        echo "ML_STARTUP: s3_model_download_failed attempting_fallback_training" >&2
        
        # Try to train a new model as fallback
        echo "ML_STARTUP: fallback_training_start" >&2
        uv run python -m divvy_ml.pipelines.xgb_trainer
        
        if [ $? -eq 0 ]; then
            # Check if any model directory was created
            for dir in /app/xgb_model*; do
                if [ -d "$dir" ]; then
                    echo "ML_STARTUP: fallback_training_success dir=$dir" >&2
                    MODEL_EXISTS=true
                    break
                fi
            done
            
            if [ "$MODEL_EXISTS" = false ]; then
                echo "ML_STARTUP: fallback_training_failed no_model_directory_created" >&2
                exit 1
            fi
        else
            echo "ML_STARTUP: fallback_training_failed" >&2
            exit 1
        fi
    fi
else
    echo "ML_STARTUP: model_exists_skip_download_and_training" >&2
fi

# Create ready flag immediately - we're ready to serve predictions when called
touch /app/output/ml_ready.flag
echo "ML_STARTUP: ready_flag_created" >&2

echo "ML_STARTUP: initialization_complete starting_server" >&2

echo "ML_STARTUP: starting_http_server_with_predictor_init" >&2
uv run python -m divvy_ml.server
