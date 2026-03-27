from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, history
from app.database import init_db
from app.services.lpr_service import lpr_service

app = FastAPI(
    title="🚗 License Plate Recognition API",
    description="API nhận diện biển số xe sử dụng YOLOv8 + EasyOCR",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc"      # ReDoc UI
)

# CORS (cho phép frontend gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo DB khi startup
@app.on_event("startup")
def startup():
    init_db()
    print("🚀 LPR API đã sẵn sàng!")

# Include routers
app.include_router(predict.router)
app.include_router(history.router)

# Health check
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "model": "loaded"}

# Model info
@app.get("/api/v1/model-info", tags=["System"])
def model_info():
    return lpr_service.get_model_info()
