import boto3
import pandas as pd
import os
import io
import logging
from typing import List, Optional
from divvy_ml.config import ml_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self, region: str = None):
        if region is None:
            region = ml_config.aws_region
        self.s3_client = boto3.client('s3', region_name=region)
        logger.info(f"s3_client_init region={region}")
    
    def upload_file(self, local_path: str, bucket_name: str, s3_key: str) -> bool:
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            file_size = os.path.getsize(local_path)
            logger.info(f"uploading file={local_path} bucket={bucket_name} key={s3_key} size_bytes={file_size}")
            
            self.s3_client.upload_file(local_path, bucket_name, s3_key)
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            
            logger.info(f"upload_success key={s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"upload_failed file={local_path} error={e}")
            return False
    
    def download_file(self, bucket_name: str, s3_key: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            logger.info(f"downloading file bucket={bucket_name} key={s3_key} local_path={local_path}")
            
            self.s3_client.download_file(bucket_name, s3_key, local_path)
            
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                logger.info(f"download_success key={s3_key} size_bytes={file_size}")
                return True
            else:
                logger.error(f"download_failed file_not_found_after_download key={s3_key}")
                return False
                
        except Exception as e:
            logger.error(f"download_failed key={s3_key} error={e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = "", suffix: str = "") -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            
            if suffix:
                objects = [obj for obj in objects if obj.endswith(suffix)]
            
            logger.info(f"list_objects bucket={bucket_name} prefix={prefix} suffix={suffix} count={len(objects)}")
            return objects
        except Exception as e:
            logger.error(f"list_failed bucket={bucket_name} prefix={prefix} error={e}")
            return []
    
    def list_csv_objects(self, bucket_name: str, prefix: str = "") -> List[str]:
        return self.list_objects(bucket_name, prefix, '.csv')
    
