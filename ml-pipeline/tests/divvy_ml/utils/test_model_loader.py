import pytest
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from divvy_ml.utils.model_loader import (
    find_local_model_directory,
    download_latest_model_from_s3,
    get_model_path,
    _validate_model_directory
)


class TestFindLocalModelDirectory:
    def test_finds_existing_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "xgb_model_01-01-24")
            os.makedirs(model_dir)
            
            result = find_local_model_directory(temp_dir)
            assert result == model_dir
    
    def test_finds_most_recent_when_multiple_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_model = os.path.join(temp_dir, "xgb_model_01-01-24")
            new_model = os.path.join(temp_dir, "xgb_model_02-01-24")
            os.makedirs(old_model)
            os.makedirs(new_model)
            
            result = find_local_model_directory(temp_dir)
            assert result == new_model
    
    def test_returns_none_when_no_model_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = find_local_model_directory(temp_dir)
            assert result is None
    
    @patch('glob.glob')
    def test_handles_glob_pattern_correctly(self, mock_glob):
        mock_glob.return_value = []
        
        find_local_model_directory("/test")
        
        mock_glob.assert_called_once_with("/test/xgb_model*")


class TestDownloadLatestModelFromS3:
    @patch('divvy_ml.utils.model_loader.S3Client')
    @patch('divvy_ml.utils.model_loader.ml_config')
    @patch('os.makedirs')
    def test_downloads_latest_model_successfully(self, mock_makedirs, mock_config, mock_s3_class):
        mock_config.aws_region = 'us-east-1'
        mock_config.model_bucket = 'test-bucket'
        
        mock_s3 = Mock()
        mock_s3_class.return_value = mock_s3
        mock_s3.list_objects.return_value = [
            'xgb_model_01-01-24/model.joblib',
            'xgb_model_01-01-24/preprocessor.joblib',
            'xgb_model_02-01-24/model.joblib',
            'xgb_model_02-01-24/preprocessor.joblib'
        ]
        mock_s3.download_file.return_value = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_latest_model_from_s3(temp_dir)
            
            expected_path = os.path.join(temp_dir, 'xgb_model_02-01-24')
            assert result == expected_path
            
            mock_s3.list_objects.assert_called_once_with('test-bucket', prefix='xgb_model')
            assert mock_s3.download_file.call_count == 2
    
    @patch('divvy_ml.utils.model_loader.S3Client')
    @patch('divvy_ml.utils.model_loader.ml_config')
    def test_returns_none_when_no_objects_in_s3(self, mock_config, mock_s3_class):
        mock_config.aws_region = 'us-east-1'
        mock_config.model_bucket = 'test-bucket'
        
        mock_s3 = Mock()
        mock_s3_class.return_value = mock_s3
        mock_s3.list_objects.return_value = []
        
        result = download_latest_model_from_s3('/test')
        assert result is None
    
    @patch('divvy_ml.utils.model_loader.S3Client')
    @patch('divvy_ml.utils.model_loader.ml_config')
    def test_returns_none_when_no_valid_directories(self, mock_config, mock_s3_class):
        mock_config.aws_region = 'us-east-1'
        mock_config.model_bucket = 'test-bucket'
        
        mock_s3 = Mock()
        mock_s3_class.return_value = mock_s3
        mock_s3.list_objects.return_value = ['file_without_directory.txt']
        
        result = download_latest_model_from_s3('/test')
        assert result is None
    
    @patch('divvy_ml.utils.model_loader.S3Client')
    @patch('divvy_ml.utils.model_loader.ml_config')
    @patch('os.makedirs')
    def test_returns_none_when_download_fails(self, mock_makedirs, mock_config, mock_s3_class):
        mock_config.aws_region = 'us-east-1'
        mock_config.model_bucket = 'test-bucket'
        
        mock_s3 = Mock()
        mock_s3_class.return_value = mock_s3
        mock_s3.list_objects.return_value = ['xgb_model_01-01-24/model.joblib']
        mock_s3.download_file.return_value = False
        
        result = download_latest_model_from_s3('/test')
        assert result is None
    
    @patch('divvy_ml.utils.model_loader.S3Client')
    @patch('divvy_ml.utils.model_loader.ml_config')
    def test_handles_s3_exception(self, mock_config, mock_s3_class):
        mock_config.aws_region = 'us-east-1'
        mock_config.model_bucket = 'test-bucket'
        
        mock_s3_class.side_effect = Exception("S3 connection failed")
        
        result = download_latest_model_from_s3('/test')
        assert result is None


class TestValidateModelDirectory:
    def test_validates_complete_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            required_files = [
                'model.joblib',
                'preprocessor.joblib',
                'feature_columns.joblib',
                'label_encoders.joblib',
                'metadata.json'
            ]
            
            for filename in required_files:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write('test')
            
            result = _validate_model_directory(temp_dir)
            assert result is True
    
    def test_fails_when_missing_required_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'model.joblib')
            with open(file_path, 'w') as f:
                f.write('test')
            
            result = _validate_model_directory(temp_dir)
            assert result is False
    
    def test_fails_when_directory_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _validate_model_directory(temp_dir)
            assert result is False


class TestGetModelPath:
    @patch('divvy_ml.utils.model_loader.find_local_model_directory')
    @patch('divvy_ml.utils.model_loader._validate_model_directory')
    def test_returns_local_path_when_valid_local_exists(self, mock_validate, mock_find_local):
        mock_find_local.return_value = '/app/xgb_model_01-01-24'
        mock_validate.return_value = True
        
        result = get_model_path('/app')
        assert result == '/app/xgb_model_01-01-24'
        
        mock_find_local.assert_called_once_with('/app')
        mock_validate.assert_called_once_with('/app/xgb_model_01-01-24')
    
    @patch('divvy_ml.utils.model_loader.find_local_model_directory')
    @patch('divvy_ml.utils.model_loader._validate_model_directory')
    @patch('divvy_ml.utils.model_loader.download_latest_model_from_s3')
    def test_downloads_from_s3_when_no_valid_local(self, mock_download, mock_validate, mock_find_local):
        mock_find_local.return_value = None
        mock_download.return_value = '/app/xgb_model_02-01-24'
        mock_validate.side_effect = [True]
        
        result = get_model_path('/app')
        assert result == '/app/xgb_model_02-01-24'
        
        mock_find_local.assert_called_once_with('/app')
        mock_download.assert_called_once_with('/app')
        mock_validate.assert_called_once_with('/app/xgb_model_02-01-24')
    
    @patch('divvy_ml.utils.model_loader.find_local_model_directory')
    @patch('divvy_ml.utils.model_loader._validate_model_directory')
    @patch('divvy_ml.utils.model_loader.download_latest_model_from_s3')
    def test_downloads_from_s3_when_local_invalid(self, mock_download, mock_validate, mock_find_local):
        mock_find_local.return_value = '/app/xgb_model_invalid'
        mock_download.return_value = '/app/xgb_model_02-01-24'
        mock_validate.side_effect = [False, True]
        
        result = get_model_path('/app')
        assert result == '/app/xgb_model_02-01-24'
        
        mock_validate.assert_any_call('/app/xgb_model_invalid')
        mock_validate.assert_any_call('/app/xgb_model_02-01-24')
        mock_download.assert_called_once_with('/app')
    
    @patch('divvy_ml.utils.model_loader.find_local_model_directory')
    @patch('divvy_ml.utils.model_loader._validate_model_directory')
    @patch('divvy_ml.utils.model_loader.download_latest_model_from_s3')
    def test_returns_none_when_all_options_fail(self, mock_download, mock_validate, mock_find_local):
        mock_find_local.return_value = None
        mock_download.return_value = None
        
        result = get_model_path('/app')
        assert result is None
        
        mock_find_local.assert_called_once_with('/app')
        mock_download.assert_called_once_with('/app')
        mock_validate.assert_not_called()
    
    @patch('divvy_ml.utils.model_loader.find_local_model_directory')
    @patch('divvy_ml.utils.model_loader._validate_model_directory')
    @patch('divvy_ml.utils.model_loader.download_latest_model_from_s3')
    def test_returns_none_when_downloaded_model_invalid(self, mock_download, mock_validate, mock_find_local):
        mock_find_local.return_value = None
        mock_download.return_value = '/app/xgb_model_invalid'
        mock_validate.return_value = False
        
        result = get_model_path('/app')
        assert result is None
        
        mock_validate.assert_called_once_with('/app/xgb_model_invalid')

