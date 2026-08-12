# Wastra AI - REST API Documentation

Detailed specifications for endpoints across Backend (Golang) and ML Service (FastAPI).

## Base URLs
- Backend Gateway: `http://localhost:8080/api/v1`
- ML Service (Internal): `http://localhost:8000/api/v1`

---

## Auth Endpoints (Backend)
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/refresh`

## Scan Endpoints (Backend)
- `POST /scan` (Multipart File + Bearer JWT)
- `GET /scan/history`

## Motif Endpoints (Backend)
- `GET /motifs`
- `GET /motifs/:id`

## ML Service Internal API
- `POST /predict` (Multipart Image File)
- `GET /health`
