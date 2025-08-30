import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from divvy_ml.utils.s3_client import S3Client
import io

class TestS3Client:
    
    def test_init(self):
        client = S3Client('us-east-2')
        assert client.s3_client is not None

    def test_upload_file_success(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024):
                result = client.upload_file('/tmp/test.csv', 'bucket', 'key')
                assert result is True
                mock_s3.upload_file.assert_called_once()
                mock_s3.head_object.assert_called_once()

    def test_upload_file_not_found(self):
        with patch('boto3.client') as mock_boto:
            mock_boto.return_value = Mock()
            client = S3Client()
            
            with patch('os.path.exists', return_value=False):
                result = client.upload_file('/tmp/missing.csv', 'bucket', 'key')
                assert result is False

    def test_upload_file_failure(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_s3.upload_file.side_effect = Exception("Upload failed")
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            
            with patch('os.path.exists', return_value=True):
                result = client.upload_file('/tmp/test.csv', 'bucket', 'key')
                assert result is False

    def test_list_objects_success(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_s3.list_objects_v2.return_value = {
                'Contents': [
                    {'Key': 'file1.csv'},
                    {'Key': 'file2.txt'},
                    {'Key': 'dir/file3.csv'}
                ]
            }
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            objects = client.list_objects('bucket', 'prefix')
            assert len(objects) == 3
            assert 'file1.csv' in objects

    def test_list_objects_with_suffix(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_s3.list_objects_v2.return_value = {
                'Contents': [
                    {'Key': 'file1.csv'},
                    {'Key': 'file2.txt'},
                    {'Key': 'dir/file3.csv'}
                ]
            }
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            objects = client.list_objects('bucket', 'prefix', '.csv')
            assert len(objects) == 2
            assert all(obj.endswith('.csv') for obj in objects)

    def test_list_objects_failure(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_s3.list_objects_v2.side_effect = Exception("List failed")
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            objects = client.list_objects('bucket')
            assert objects == []

    def test_download_file_success(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024):
                result = client.download_file('bucket', 'key', '/tmp/file')
                assert result is True
                mock_s3.download_file.assert_called_once()

    def test_download_file_failure(self):
        with patch('boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_s3.download_file.side_effect = Exception("Download failed")
            mock_boto.return_value = mock_s3
            
            client = S3Client()
            result = client.download_file('bucket', 'key', '/tmp/file')
            assert result is False

