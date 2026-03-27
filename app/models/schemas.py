from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class PlateDetection(BaseModel):
    plate_text: str
    confidence: float
    bbox: BoundingBox

class PredictResponse(BaseModel):
    filename: str
    detections: List[PlateDetection]
    total_plates: int
    process_time_ms: float

class HistoryRecord(BaseModel):
    id: int
    filename: str
    plate_text: str
    confidence: float
    created_at: datetime

class ModelInfoResponse(BaseModel):
    model_name: str
    framework: str
    input_size: int
    classes: List[str]
    status: str