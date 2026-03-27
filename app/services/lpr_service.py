import cv2
import numpy as np
import time
import easyocr
from ultralytics import YOLO
from pathlib import Path

# Khởi tạo model 1 lần duy nhất khi server start
MODEL_PATH = Path("weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

class LPRService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
        self.model = YOLO(str(MODEL_PATH))
        self.classes = self.model.names  # dict {0: 'license_plate', ...}
        print(f"✅ Loaded model: {MODEL_PATH}")

    def predict(self, image_bytes: bytes, filename: str) -> dict:
        start = time.time()

        # Decode ảnh từ bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO detect biển số
        results = self.model(img, conf=0.5, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Crop vùng biển số
                plate_crop = img[y1:y2, x1:x2]

                # OCR đọc text
                plate_text = self._ocr(plate_crop)

                detections.append({
                    "plate_text": plate_text,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

        process_time = (time.time() - start) * 1000
        return {
            "filename": filename,
            "detections": detections,
            "total_plates": len(detections),
            "process_time_ms": round(process_time, 2)
        }

    def _ocr(self, plate_img: np.ndarray) -> str:
        # Preprocess: grayscale + resize
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        results = reader.readtext(gray, detail=0, paragraph=True)
        text = " ".join(results).upper().strip()
        return text if text else "UNKNOWN"

    def get_model_info(self) -> dict:
        return {
            "model_name": "YOLOv8 LPR",
            "framework": "Ultralytics YOLOv8",
            "input_size": 640,
            "classes": list(self.classes.values()),
            "status": "ready"
        }

# Singleton instance
lpr_service = LPRService()