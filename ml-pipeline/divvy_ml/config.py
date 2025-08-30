import os
import logging

logger = logging.getLogger(__name__)


class MLConfig:
    """Configuration management for ML pipeline with environment variables."""
    
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-2')
        self.sagemaker_bucket = os.getenv('SAGEMAKER_S3_BUCKET', 'amazon-sagemaker-296171619255-us-east-1-82f0818aa7dc')
        self.divvy_backup_bucket = os.getenv('DIVVY_BACKUP_S3_BUCKET', 'divvy-backups-nu-434')
        self.model_bucket = os.getenv('MODEL_S3_BUCKET', 'divvy-bike-availability-trained-models')
        self.sagemaker_base_path = os.getenv('SAGEMAKER_BASE_PATH', 'dzd_3kxgvr5h9xz5rb/4bxwt7sj9u3wvb/raw/')
        self.incremental_path = os.getenv('S3_INCREMENTAL_PATH', 'incremental-v2/')
        self.ml_port = int(os.getenv('ML_PORT', '5000'))
        
        # Derived paths
        self.stations_csv_path = f'{self.divvy_backup_bucket}/stations.csv'
        
        logger.info(f"config_loaded aws_region={self.aws_region} "
                   f"sagemaker_bucket={self.sagemaker_bucket} "
                   f"backup_bucket={self.divvy_backup_bucket} "
                   f"model_bucket={self.model_bucket}")
    
    def get_s3_config_dict(self):
        """Return configuration as dict for backward compatibility."""
        return {
            'SAGEMAKER_BUCKET': self.sagemaker_bucket,
            'DIVVY_BACKUP_BUCKET': self.divvy_backup_bucket,
            'MODEL_BUCKET': self.model_bucket,
            'SAGEMAKER_BASE_PATH': self.sagemaker_base_path,
            'DIVVY_INCREMENTAL_PATH': self.incremental_path,
            'AWS_REGION': self.aws_region
        }


# Global config instance
ml_config = MLConfig()
