<<<<<<< HEAD
# Batik-AI-
=======
# 🌿 Wastra AI

Monorepo platform pengenalan motif batik berbasis Deep Learning dengan arsitektur Microservices modern.

## 📁 Repository Architecture

```
wastra-ai/
├── apps/
│   ├── mobile/         # Mobile App (Flutter - Feature-First Architecture)
│   ├── backend/        # REST API Gateway (Golang - Uncle Bob Clean Architecture)
│   └── ml-service/     # Deep Learning Inference (Python FastAPI + TensorFlow)
├── training/           # ML Training Pipelines & Notebooks
├── datasets/           # Raw & Processed Datasets
├── docs/               # Architecture & API Documentation
├── deployment/         # Docker & Deployment Infrastructure
├── scripts/            # Automation & Utility Scripts
├── .github/            # CI/CD Workflows
└── docker-compose.yml  # Multi-Container Orchestration
```

## 🚀 Quick Setup

```bash
# Clone & Start Services via Docker Compose
docker-compose up --build -d
```
>>>>>>> 0692da1 (feat: initial commit for Batik AI project with refined EDA notebook)
