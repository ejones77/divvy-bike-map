# Divvy Bike Map

Cloud service for predicting the availability of Divvy bikes in Chicago using machine learning.

## Components

### 1. API Service (Go/Gin)

The main web server handling:
- **Web UI**: Interactive map showing bike station availability
- **REST API**: JSON endpoints for station data
- **Data Collection**: Fetches Divvy station status every 15 minutes
- **Prediction Orchestration**: Calls ML service every 2 hours for predictions

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Map homepage |
| `/stations` | GET | Station list with current availability |
| `/predictions` | GET | Station list with predicted availability |
| `/api/stations/json` | GET | JSON station data |
| `/api/refresh` | POST | Manually trigger data refresh |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### 2. ML Service (Python/Flask)

Machine learning inference server:
- **XGBoost Model**: Predicts bike availability classes
- **Model Loading**: Downloads model from S3 on startup (or uses local fallback)
- **Prediction Caching**: 15-minute cache to reduce compute

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Run inference, returns JSON predictions |
| `/health` | GET | Health check (predictor status) |
| `/status` | GET | Detailed prediction status |
| `/metrics` | GET | Prometheus metrics |

### 3. Monitoring Stack

**Prometheus**
- Scrapes metrics from API and ML services every 15 seconds
- Stores time-series data for analysis

**Grafana**
- Pre-configured dashboard with:
  - API request rate
  - ML prediction counts
  - Prediction duration over time
- Default credentials: `admin` / `admin` (or `GRAFANA_ADMIN_PASSWORD` env var)

## Data Flow

```
1. Data Collection (every 15 min)
   Divvy GBFS API → API Service → Postgres DB
   
2. Prediction Generation (every 2 hours)
   API Service → POST /predict → ML Service
   ML Service → Load features from DB → XGBoost inference
   ML Service → Return predictions → API Service → Store in DB
   
3. User Request
   Browser → API Service → Query DB → Render map/HTML
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DB_URL=postgresql://user:password@host:PORT/dbname

# AWS (for model storage)
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MODEL_S3_BUCKET=your-trained-models

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_secure_password

# Optional timing overrides
DATA_COLLECTION_INTERVAL_MIN=15    # How often to fetch station data
PREDICTION_INTERVAL_HOURS=2        # How often to regenerate predictions
ML_REQUEST_TIMEOUT_MIN=5           # Timeout for ML service calls
```

## Dependencies

- **uv** (Python 3.12) — ML pipeline package management
- **Go** (1.24+) — API server
- **Docker & Docker Compose** — Container orchestration
- **Make** — Local development commands
- **AWS CLI** — For S3 model uploads (optional)

## Quick Start (Local Development)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd divvy-bike-map

# 2. Set up environment
cp .env.example .env
# Edit .env with your credentials

# 3. Start all services
make up

# 4. View the map
open http://localhost

# 5. View logs
make logs

# 6. Stop services
make down
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make up` | Start all services (dev mode with volume mounts) |
| `make up-dev` | Same as `make up` |
| `make up-prod` | Start production containers (no volume mounts) |
| `make down` | Stop all services |
| `make logs` | Tail logs from all services |
| `make db-shell` | Connect to Postgres CLI |
| `make db-refresh` | Manually trigger station data refresh |
| `make clean` | Stop services and remove volumes |
| `make rebuild` | Rebuild API container and restart |
| `make rebuild-all` | Rebuild all containers and restart |

## Training a New Model

### First-Time Setup: Data Collection

> ⚠️ **WARNING: Data Volume**
> 
> The data collection process fetches availability for **~800 stations every 15 minutes**. This accumulates **~77,000 rows per day** (~2.3M rows per month). Your Postgres database will grow quickly — plan your storage accordingly.

> Realistically, for this project to live at a reasonable cost, it needs retention/archival policies to dump availability data after training every 2-weeks or so. This modeling method is probably best on shorter horizons anyway.

Before you can train a model, you need historical station availability data. The model trainer expects at least **1 million rows** for decent predictions (roughly **2 weeks of collection**).

**Data collection timeline:**
| Duration | Rows (approx) | Model Quality |
|----------|---------------|---------------|
| 1 day | ~77K | ❌ Not enough |
| 1 week | ~540K | ⚠️ Marginal |
| 2 weeks | ~1.1M | ✅ Minimum viable |
| 1 month | ~2.3M | ✅ Great |
| 3 months | ~7M | ✅ Excellent |

**To start collecting data:**

```bash
# 1. Set up your .env with DB_URL pointing to Postgres
# 2. Start ONLY the API service (no ML needed yet)
make up

# 3. The API will automatically:
#    - Fetch station data from Divvy every 15 minutes
#    - Store it in the public.station_availability table
#    - Run indefinitely until you stop it

# 4. Monitor data accumulation
docker-compose exec postgres psql -U divvy_user -d divvy -c \
  "SELECT COUNT(*) as rows, MIN(recorded_at), MAX(recorded_at) FROM public.station_availability;"

# 5. Let it run for at least 2 weeks before training
```

### Training the Model

Once you have enough data collected:

```bash
cd ml-pipeline

# Install dependencies
uv sync

# Train model (pulls last 30 days from DB by default)
uv run --env-file .env python divvy_ml/pipelines/xgb_trainer.py

# Output: ml-pipeline/xgb_model_<MM-DD-YY>/
```

**Training parameters** (in `xgb_trainer.py`):
- `days_back=30` — How many days of historical data to use
- `tune_hyperparams=True` — Run hyperparameter search (slower but better)
- `n_features=30` — Number of features after selection

### Uploading Model to S3

For production, upload the trained model to S3 so the ML service can download it on startup:

```bash
# Upload entire model directory to S3
aws s3 cp ml-pipeline/xgb_model_<DATE>/ s3://$MODEL_S3_BUCKET/xgb_model_<DATE>/ --recursive

# The ML service will automatically find and download the latest model on startup
```

### Data Retention (Recommended)

To prevent your database from growing indefinitely, consider adding a cleanup job:

```sql
-- Delete data older than 90 days (run periodically via cron or pg_cron)
DELETE FROM public.station_availability 
WHERE recorded_at < NOW() - INTERVAL '90 days';
```

Or set up a partition strategy for the `station_availability` table if you need to keep historical data but want faster queries.

## Production Deployment (EC2)

The deployment is automated via GitHub Actions, but here's the manual process:

### Prerequisites on EC2
- Docker and Docker Compose installed
- AWS CLI configured
- Access to ECR registry

### Deploy Steps

```bash
# 1. SSH to EC2 instance

# 2. Navigate to app directory
cd /opt/divvy-bike-availability

# 3. Pull latest code
git pull origin main

# 4. Update .env file if needed

# 5. Pull and start containers
docker-compose pull
docker-compose up -d

# 6. Verify services are running
docker-compose ps
docker-compose logs -f
```

### Automated Deployment

The `scripts/deploy.sh` script handles:
1. Installing Docker/Docker Compose/AWS CLI if missing
2. Cloning/updating the repository
3. Configuring AWS credentials
4. Creating the `.env` file from secrets
5. Updating `docker-compose.yml` with ECR image references
6. Starting services
7. Setting up cron jobs for auto-restart and cleanup

### Cron Jobs (set up automatically)
- `@reboot` — Auto-start Docker services on instance reboot
- `0 2 * * 0` — Weekly Docker system cleanup (Sundays at 2 AM)

## Monitoring Access

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | `http://<host>:3000` | admin / GRAFANA_ADMIN_PASSWORD |
| Prometheus | `http://<host>:9090` | (no auth) |

### Grafana Dashboard

The pre-configured dashboard shows:
- **API Request Rate**: Requests per second to the API service
- **ML Prediction Requests**: Total count of prediction generations
- **ML Prediction Duration**: Time series of inference latency

### Adding Custom Metrics

Both services expose `/metrics` endpoints with Prometheus format:

**API Metrics** (Go/Gin):
- Standard HTTP metrics via `promhttp`

**ML Metrics** (Python/Flask):
- `ml_prediction_requests_total` — Counter of prediction requests
- `ml_prediction_duration_seconds` — Histogram of prediction times

## Troubleshooting

### Services won't start
```bash
# Check logs for specific service
docker-compose logs api
docker-compose logs ml

# ML service health check failing? Model might not have loaded
docker-compose logs ml | grep ML_STARTUP
```

### ML service stuck in "not ready"
```bash
# Check if model exists locally
ls ml-pipeline/xgb_model*/

# Or check if S3 download succeeded
docker-compose logs ml | grep s3
```

### Database connection issues
```bash
# Verify DB_URL is set
echo $DB_URL

# Test connection
docker-compose exec api wget -qO- http://localhost:8080/health
```

### Reset everything
```bash
make clean
make rebuild-all
```

## Project Structure

```
divvy-bike-map/
├── api/                    # Go API server
│   ├── cmd/server/         # Main entrypoint
│   ├── internal/           # Server logic, handlers, config
│   ├── migrations/         # SQL migrations
│   ├── static/             # JS/CSS assets
│   └── templates/          # HTML templates
├── ml-pipeline/            # Python ML service
│   ├── divvy_ml/           # Core ML code
│   │   ├── pipelines/      # Training & inference
│   │   ├── utils/          # DB, S3, model loading
│   │   └── server.py       # Flask inference server
│   ├── tests/              # Unit tests
│   └── xgb_model/          # Trained model artifacts
├── monitoring/             # Prometheus & Grafana config
│   ├── prometheus.yml
│   └── grafana/
│       ├── datasources/
│       └── dashboards/
├── scripts/
│   └── deploy.sh           # EC2 deployment script
├── docker-compose.yml      # Production config
├── docker-compose.dev.yml  # Development config
└── Makefile                # Dev commands
```

## Tests

### API Tests (Go)
```bash
cd api
go test ./internal/...
```

### ML Pipeline Tests (Python)
```bash
cd ml-pipeline
uv run pytest tests/
```

### Frontend Tests (Vitest)
```bash
cd api/static
npm test
```
