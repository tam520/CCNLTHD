from fastapi import APIRouter, Query
from app.database import get_history, search_plate

router = APIRouter(prefix="/api/v1", tags=["History"])

@router.get("/history", summary="Lịch sử nhận diện biển số")
def get_detection_history(limit: int = Query(default=20, ge=1, le=100)):
    return get_history(limit=limit)

@router.get("/search", summary="Tìm kiếm biển số")
def search_license_plate(q: str = Query(..., min_length=2, description="Chuỗi biển số cần tìm")):
    results = search_plate(q)
    return {"query": q, "results": results, "total": len(results)}