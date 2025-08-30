import psycopg2
import pandas as pd
import os
from typing import Optional
import logging
from pathlib import Path
from datetime import timedelta, timezone

logger = logging.getLogger(__name__)

def _load_local_files(data_dir: str):
    p = Path(data_dir)
    stations_path = p / "stations.json"
    sa_path = p / "station_availability.json"
    if not stations_path.exists() or not sa_path.exists():
        raise FileNotFoundError(f"Local data files not found in {data_dir}")
    stations_df = pd.read_json(stations_path)
    sa_df = pd.read_json(sa_path)
    sa_df['recorded_at'] = pd.to_datetime(sa_df['recorded_at'])
    return stations_df, sa_df


def _filter_by_time(df: pd.DataFrame, time_col: str, delta: timedelta):
    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(seconds=delta.total_seconds())
    return df[df[time_col] >= cutoff].copy()

class DatabaseClient:
    def __init__(self, db_url: str = None):
        # Prefer explicit db_url, fall back to DB_URL env var.
        # If LOCAL_DATA_DIR is set we won't require a DB_URL for training with local files.
        self.db_url = db_url or os.environ.get('DB_URL')
        self.local_data_dir = os.environ.get('LOCAL_DATA_DIR')
        if not self.db_url and not self.local_data_dir:
            raise ValueError("DB_URL environment variable required unless LOCAL_DATA_DIR is set")
    
    def get_current_snapshot(self) -> pd.DataFrame:
        """Load only the latest record per station for efficient inference"""
        if os.environ.get('LOCAL_DATA_DIR'):
            stations_df, sa_df = _load_local_files(os.environ['LOCAL_DATA_DIR'])
            # latest per station
            idx = sa_df.groupby('station_id')['recorded_at'].idxmax()
            df = sa_df.loc[idx].sort_values('station_id').reset_index(drop=True)
            df = df.merge(stations_df[['station_id','name','lat','lon','capacity']],
                        on='station_id', how='left')
            return df
        else:
            query = """
            WITH latest_per_station AS (
                SELECT 
                    station_id,
                    MAX(recorded_at) as latest_recorded_at
                FROM station_availability 
                WHERE recorded_at >= NOW() - INTERVAL '2 hours'
                GROUP BY station_id
            )
            SELECT 
                sa.station_id,
                sa.num_bikes_available,
                sa.num_docks_available,
                sa.is_installed,
                sa.is_renting,
                sa.is_returning,
                sa.last_reported,
                sa.recorded_at,
                s.name,
                s.lat,
                s.lon,
                s.capacity
            FROM station_availability sa
            JOIN latest_per_station lps ON sa.station_id = lps.station_id 
                AND sa.recorded_at = lps.latest_recorded_at
            JOIN stations s ON sa.station_id = s.station_id
            ORDER BY sa.station_id
            """
            
            try:
                logger.info("loading_current_snapshot")
                conn = psycopg2.connect(self.db_url)
                df = pd.read_sql(query, conn)
                conn.close()
                
                logger.info(f"loaded_snapshot rows={len(df)} stations={df['station_id'].nunique()}")
                return df
                
            except Exception as e:
                logger.error(f"snapshot_load_error={e}")
                raise

    def get_recent_availability_data(self, hours_back: int = 2) -> pd.DataFrame:
        """Load recent availability data directly from PostgreSQL"""
        if os.environ.get('LOCAL_DATA_DIR'):
            stations_df, sa_df = _load_local_files(os.environ['LOCAL_DATA_DIR'])
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=hours_back)
            df = sa_df[sa_df['recorded_at'] >= cutoff].sort_values('recorded_at', ascending=False)
            df = df.merge(stations_df[['station_id','name','lat','lon','capacity']],
                        on='station_id', how='left')
            return df.reset_index(drop=True)
        else:
            query = """
            SELECT 
                sa.station_id,
                sa.num_bikes_available,
                sa.num_docks_available,
                sa.is_installed,
                sa.is_renting,
                sa.is_returning,
                sa.last_reported,
                sa.recorded_at,
                s.name,
                s.lat,
                s.lon,
                s.capacity
            FROM station_availability sa
            JOIN stations s ON sa.station_id = s.station_id
            WHERE sa.recorded_at >= NOW() - INTERVAL '%s hours'
            ORDER BY sa.recorded_at DESC
            """
            
            try:
                logger.info(f"loading_recent_data hours_back={hours_back}")
                conn = psycopg2.connect(self.db_url)
                df = pd.read_sql(query, conn, params=(hours_back,))
                conn.close()
                
                logger.info(f"loaded_data rows={len(df)} stations={df['station_id'].nunique()}")
                return df
                
            except Exception as e:
                logger.error(f"db_load_error={e}")
                raise
    
    def get_stations_metadata(self) -> pd.DataFrame:
        """Load station metadata"""
        if os.environ.get('LOCAL_DATA_DIR'):
            stations_df, _ = _load_local_files(os.environ['LOCAL_DATA_DIR'])
            return stations_df[['station_id','name','lat','lon','capacity']]
        else:
            query = "SELECT station_id, name, lat, lon, capacity FROM stations"
        
            try:
                conn = psycopg2.connect(self.db_url)
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            except Exception as e:
                logger.error(f"stations_load_error={e}")
                raise
    
    def get_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load historical data for training."""
        if os.environ.get('LOCAL_DATA_DIR'):
            stations_df, sa_df = _load_local_files(os.environ['LOCAL_DATA_DIR'])
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days_back)
            df = sa_df[sa_df['recorded_at'] >= cutoff].sort_values('recorded_at')
            df = df.merge(stations_df[['station_id','name','lat','lon','capacity']],
                        on='station_id', how='left')
            return df.reset_index(drop=True)
        else:
            query = """
            SELECT 
                sa.station_id,
                sa.num_bikes_available,
                sa.num_docks_available,
                sa.is_installed,
                sa.is_renting,
                sa.is_returning,
                sa.last_reported,
                sa.recorded_at,
                s.name,
                s.lat,
                s.lon,
                s.capacity
            FROM station_availability sa
            LEFT JOIN stations s ON sa.station_id = s.station_id
            WHERE sa.recorded_at >= NOW() - INTERVAL '%s days'
            ORDER BY sa.recorded_at
            """
            
            try:
                logger.info(f"loading_training_data days_back={days_back}")
                conn = psycopg2.connect(self.db_url)
                df = pd.read_sql(query, conn, params=(days_back,))
                conn.close()
                
                logger.info(f"loaded_training_data rows={len(df)} stations={df['station_id'].nunique()}")
                return df
                
            except Exception as e:
                logger.error(f"training_data_load_error={e}")
                raise
    
    def get_availability_data(self, hours_back: int = 2, inference_mode: bool = False) -> pd.DataFrame:
        """Unified method for loading data - snapshot for inference, historical for training"""
        if inference_mode:
            return self.get_current_snapshot()
        else:
            return self.get_recent_availability_data(hours_back)
