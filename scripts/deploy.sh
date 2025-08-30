#!/bin/bash
set -e

# Deployment script for Divvy Bike Map
# This script is executed on the EC2 instance via AWS SSM

echo "=== Starting Divvy Bike Map Deployment ==="
echo "ECR Registry: $ECR_REGISTRY"
echo "Image Tag: $IMAGE_TAG"
echo "Repository: $REPOSITORY"

# Function to install dependencies if not present
install_dependencies() {
    echo "=== Checking and installing dependencies ==="
    
    # Install Docker if not present
    if ! command -v docker >/dev/null; then
        echo "Installing Docker..."
        sudo apt-get update -qq
        sudo apt-get install -y docker.io git curl unzip
        sudo systemctl enable --now docker
        sudo usermod -aG docker $USER
    fi
    
    # Install Docker Compose if not present
    if ! command -v docker-compose >/dev/null; then
        echo "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    fi
    
    # Install AWS CLI if not present
    if ! command -v aws >/dev/null; then
        echo "Installing AWS CLI..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip -q awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
}

# Function to setup application directory
setup_app_directory() {
    echo "=== Setting up application directory ==="
    APP_DIR="/opt/divvy-bike-availability"
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
    cd $APP_DIR
    
    # Clone or update repository
    if [ -d .git ]; then
        echo "Updating existing repository..."
        git fetch origin
        git reset --hard origin/main
    else
        echo "Cloning repository..."
        git clone https://github.com/$GITHUB_REPOSITORY .
    fi
}

# Function to configure AWS credentials
configure_aws() {
    echo "=== Configuring AWS credentials ==="
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    aws configure set default.region $AWS_REGION
    
    echo "Logging into ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
}

# Function to create environment file
create_env_file() {
    echo "=== Creating environment file ==="
    cat > .env << EOF
DB_URL=$PRODUCTION_DB_URL
AWS_REGION=$AWS_REGION
MODEL_S3_BUCKET=$MODEL_S3_BUCKET
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASSWORD
ML_REQUEST_TIMEOUT_MIN=5
ML_PORT=5000
DATA_COLLECTION_INTERVAL_MIN=15
PREDICTION_INTERVAL_HOURS=2
EOF
}

# Function to update docker-compose file with ECR images
update_docker_compose() {
    echo "=== Updating docker-compose.yml with ECR images ==="
    sed -i "s|build: ./api|image: $ECR_REGISTRY/$REPOSITORY:api-$IMAGE_TAG|" docker-compose.yml
    sed -i "s|build: ./ml-pipeline|image: $ECR_REGISTRY/$REPOSITORY:ml-$IMAGE_TAG|" docker-compose.yml
}

# Function to deploy application
deploy_application() {
    echo "=== Deploying application ==="
    docker-compose pull
    docker-compose up -d --remove-orphans
    
    echo "=== Cleaning up old Docker images ==="
    docker image prune -f --filter "until=24h"
    
    docker system prune -f
    
    echo "=== Setting up cron jobs ==="
    setup_cron_jobs || echo "Warning: Cron job setup failed, but deployment continues"
}

# Function to setup cron jobs (non-critical)
setup_cron_jobs() {
    echo "=== Setting up auto-restart on boot ==="
    APP_DIR="/opt/divvy-bike-availability"
    (crontab -l 2>/dev/null | grep -v "@reboot.*divvy-bike"; echo "@reboot cd $APP_DIR && docker-compose up -d") | crontab -
    
    echo "=== Setting up weekly Docker cleanup ==="
    (crontab -l 2>/dev/null | grep -v "docker system prune"; echo "0 2 * * 0 docker system prune -a -f --volumes") | crontab -
}

# Main deployment flow
main() {
    install_dependencies
    setup_app_directory
    configure_aws
    create_env_file
    update_docker_compose
    deploy_application
    
    echo "=== Deployment completed successfully ==="
}

# Run main function
main
