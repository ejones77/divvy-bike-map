import pytest
import pandas as pd
from unittest.mock import Mock, patch
import os

class TestDatabaseClient:
    
    def test_init_with_url(self):
        from divvy_ml.utils.database import DatabaseClient
        client = DatabaseClient("postgresql://user:pass@host:5432/db")
        assert client.db_url == "postgresql://user:pass@host:5432/db"
    
    def test_init_from_env(self):
        # DB_URL already mocked in conftest.py
        from divvy_ml.utils.database import DatabaseClient
        client = DatabaseClient()
        assert client.db_url == 'postgresql://test:test@test:5432/test'
    
    def test_init_no_url_raises_error(self):
        from divvy_ml.utils.database import DatabaseClient
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="DB_URL environment variable required"):
                DatabaseClient()
    
    @patch('psycopg2.connect')
    @patch('pandas.read_sql')
    def test_get_recent_availability_data(self, mock_read_sql, mock_connect, sample_availability_data):
        from divvy_ml.utils.database import DatabaseClient
        
        mock_read_sql.return_value = sample_availability_data
        
        client = DatabaseClient("postgresql://test:test@test:5432/test")
        result = client.get_recent_availability_data(hours_back=2)
        
        assert len(result) > 0
        mock_connect.assert_called_once()
        mock_read_sql.assert_called_once()
    
    @patch('psycopg2.connect')
    @patch('pandas.read_sql')
    def test_get_stations_metadata(self, mock_read_sql, mock_connect, sample_stations_data):
        from divvy_ml.utils.database import DatabaseClient
        
        mock_read_sql.return_value = sample_stations_data
        
        client = DatabaseClient("postgresql://test:test@test:5432/test")
        result = client.get_stations_metadata()
        
        assert len(result) == 3
        assert 'station_id' in result.columns
        mock_connect.assert_called_once()
