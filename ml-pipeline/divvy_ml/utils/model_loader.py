import os
import glob
import logging
from typing import Optional
from divvy_ml.config import ml_config
from divvy_ml.utils.s3_client import S3Client

logger = logging.getLogger(__name__)


def find_local_model_directory(base_path: str = "/app") -> Optional[str]:
    pattern = os.path.join(base_path, "xgb_model*")
    model_dirs = glob.glob(pattern)
    
    if model_dirs:
        # Sort to get the most recent (assuming date-based naming)
        model_dirs.sort(reverse=True)
        selected_dir = model_dirs[0]
        logger.info(f"found_local_model_directory path={selected_dir}")
        return selected_dir
    
    logger.warning(f"no_local_model_directory_found pattern={pattern}")
    return None


def download_latest_model_from_s3(local_base_path: str = "/app") -> Optional[str]:
    try:
        s3_client = S3Client(region=ml_config.aws_region)
        bucket = ml_config.model_bucket
        
        logger.info(f"s3_model_search bucket={bucket}")
        objects = s3_client.list_objects(bucket, prefix="xgb_model")
        
        if not objects:
            logger.error(f"no_models_found_in_s3 bucket={bucket}")
            return None
        
        # Find the latest model directory (assuming date-based naming)
        model_prefixes = set()
        for obj_key in objects:
            if "/" in obj_key:
                model_dir = obj_key.split("/")[0]
                model_prefixes.add(model_dir)
        
        if not model_prefixes:
            logger.error(f"no_valid_model_directories_found objects={len(objects)}")
            return None
        
        latest_model_dir = sorted(model_prefixes, reverse=True)[0]
        logger.info(f"selected_latest_model s3_dir={latest_model_dir}")
        
        local_model_path = os.path.join(local_base_path, latest_model_dir)
        os.makedirs(local_model_path, exist_ok=True)
        
        model_files = [obj for obj in objects if obj.startswith(f"{latest_model_dir}/")]
        
        if not model_files:
            logger.error(f"no_files_found_in_model_directory s3_dir={latest_model_dir}")
            return None
        
        download_success = True
        for s3_key in model_files:
            filename = os.path.basename(s3_key)
            if filename:
                local_file_path = os.path.join(local_model_path, filename)
                success = s3_client.download_file(bucket, s3_key, local_file_path)
                if not success:
                    logger.error(f"failed_to_download_file s3_key={s3_key}")
                    download_success = False
        
        if download_success:
            logger.info(f"model_download_success local_path={local_model_path} files={len(model_files)}")
            return local_model_path
        else:
            logger.error(f"model_download_failed local_path={local_model_path}")
            return None
            
    except Exception as e:
        logger.error(f"model_download_error error={e}")
        return None


def get_model_path(base_path: str = "/app") -> str:
    local_model_path = find_local_model_directory(base_path)
    if local_model_path and _validate_model_directory(local_model_path):
        logger.info(f"using_existing_local_model path={local_model_path}")
        return local_model_path
    
    # If no valid local model, try to download from S3
    logger.info("no_valid_local_model_found attempting_s3_download")
    downloaded_path = download_latest_model_from_s3(base_path)
    if downloaded_path and _validate_model_directory(downloaded_path):
        logger.info(f"using_downloaded_model path={downloaded_path}")
        return downloaded_path


def _validate_model_directory(model_path: str) -> bool:
    required_files = [
        "model.joblib",
        "preprocessor.joblib", 
        "feature_columns.joblib",
        "label_encoders.joblib",
        "metadata.json"
    ]
    
    for filename in required_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            logger.warning(f"missing_model_file path={file_path}")
            return False
    
    logger.info(f"model_directory_valid path={model_path}")
    return True
