from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.lpr_service import lpr_service
from app.database import save_detection
from app.models.schemas import PredictResponse

router = APIRouter(prefix="/api/v1", tags=["Predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@router.post("/predict", response_model=PredictResponse, summary="Nhận diện biển số từ ảnh")
async def predict_image(file: UploadFile = File(..., description="Ảnh chứa biển số xe")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Chỉ chấp nhận ảnh JPG/PNG/WEBP")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 10MB)")

    try:
        result = lpr_service.predict(image_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    # Lưu vào DB
    for det in result["detections"]:
        save_detection(file.filename, det["plate_text"], det["confidence"])

    return result