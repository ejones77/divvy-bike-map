# Makefile
.PHONY: help up up-db up-full down logs db-shell db-refresh clean up-dev up-prod rebuild rebuild-all

help:
	@echo "Available commands:"
	@echo "  make up-db     - Start PostgreSQL only (dev)"
	@echo "  make up-dev    - Start all services (dev with volume mounts)"
	@echo "  make up-prod   - Start all services (production)"
	@echo "  make up        - Start all services (alias for up-dev)"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - View logs"
	@echo "  make db-shell  - Connect to PostgreSQL"
	@echo "  make db-refresh - Manually refresh station data"
	@echo "  make clean     - Clean everything"

up-db:
	docker-compose -f docker-compose.dev.yml up -d postgres

up-dev:
	docker-compose -f docker-compose.dev.yml up -d

up-prod:
	docker-compose -f docker-compose.yml up -d

up: up-dev

up-full: up-dev

down:
	docker-compose -f docker-compose.dev.yml down

logs:
	docker-compose -f docker-compose.dev.yml logs -f

db-shell:
	docker-compose -f docker-compose.dev.yml exec postgres psql -U divvy_user -d divvy

db-test:
	docker-compose -f docker-compose.dev.yml exec postgres pg_isready -U divvy_user -d divvy

db-refresh:
	curl -X POST http://localhost:8080/api/refresh

clean:
	docker-compose -f docker-compose.dev.yml down -v
	rm -rf postgres_data/

rebuild:
	docker-compose down
	docker-compose build --no-cache api
	docker-compose -f docker-compose.dev.yml up -d

rebuild-all:
	docker-compose down
	docker-compose build --no-cache
	docker-compose -f docker-compose.dev.yml up -d