import cv2
import numpy as np
import time
import re 
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

                # Crop vùng biển số
                plate_crop = img[y1:y2, x1:x2]

                # OCR đọc text (Hứng 2 biến: text và điểm tự tin OCR)
                plate_text, ocr_conf = self._ocr(plate_crop)

                detections.append({
                    "plate_text": plate_text,
                    "confidence": ocr_conf, 
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

        process_time = (time.time() - start) * 1000
        return {
            "filename": filename,
            "detections": detections,
            "total_plates": len(detections),
            "process_time_ms": round(process_time, 2)
        }

    def _ocr(self, plate_img: np.ndarray) -> tuple[str, float]:
        # 1. Tiền xử lý Tối Giản Bậc Nhất: Chỉ chuyển xám (Bỏ hoàn toàn Resize)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Đọc OCR trực tiếp trên ảnh gốc cắt ra
        ocr_results = reader.readtext(
            gray, 
            detail=1, 
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. '
        )
        
        raw_text = ""
        confidences = []
        for bbox, text, prob in ocr_results:
            raw_text += text
            confidences.append(prob)
            
        # 3. Tính điểm tự tin trung bình
        avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        # 4. Hậu xử lý Regex
        final_text = self._format_plate_vietnam(raw_text)

        return (final_text if final_text else "UNKNOWN"), round(avg_conf, 2)
    def _format_plate_vietnam(self, text: str) -> str:
        """
        Business Logic: Tự động nắn biển xe máy với độ an toàn cao nhất.
        """
        pure_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # [CHỐT CHẶN RÁC] Nếu AI đọc dư rác sinh ra 10-11 ký tự, chủ động gọt đuôi lấy 9
        if len(pure_text) > 9:
            pure_text = pure_text[:9]
            
        if len(pure_text) < 8: 
            return pure_text
        
        # Cố định tiền tố BẮT BUỘC là 4 ký tự để tránh lỗi IndexError (Out of Range)
        prefix = pure_text[:4]
        tail = pure_text[4:]
        
        # 2. NẮN GÂN TIỀN TỐ
        h12 = prefix[:2].translate(str.maketrans('OIZBSG', '012856'))
        char_3 = prefix[2].translate(str.maketrans('01284', 'OIZBA'))
        char_4 = prefix[3] 
        formatted_prefix = f"{h12}-{char_3}{char_4}"
        
        # 3. NẮN GÂN HẬU TỐ (100% SỐ)
        tail = tail.translate(str.maketrans('OILZSBQAJGTD', '011258041610'))
        
        # 4. TRÌNH BÀY
        if len(tail) == 5:
            return f"{formatted_prefix} {tail[:3]}.{tail[3:]}"
            
        return f"{formatted_prefix} {tail}"

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
