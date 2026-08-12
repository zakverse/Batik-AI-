.PHONY: help up down restart logs build

help:
	@echo "Wastra AI Monorepo Commands:"
	@echo "  make up       - Start all docker containers"
	@echo "  make down     - Stop all docker containers"
	@echo "  make restart  - Restart containers"
	@echo "  make logs     - Tail container logs"
	@echo "  make build    - Rebuild docker images"

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

build:
	docker-compose build
