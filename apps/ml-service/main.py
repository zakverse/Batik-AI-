from fastapi import FastAPI

app = FastAPI(
    title="Wastra AI - ML Inference Service",
    description="Microservice for Batik Motif Recognition using EfficientNetB0",
    version="1.0.0"
)

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "service": "wastra-ml-service"}
