.PHONY: install dev test lint format run docker-up docker-down

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down
