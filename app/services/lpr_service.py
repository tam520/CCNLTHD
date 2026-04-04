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
                # Đã bỏ lấy yolo_conf theo yêu cầu

                # Crop vùng biển số
                plate_crop = img[y1:y2, x1:x2]

                # OCR đọc text (Hứng 2 biến: text và điểm tự tin OCR)
                plate_text, ocr_conf = self._ocr(plate_crop)

                detections.append({
                    "plate_text": plate_text,
                    "confidence": ocr_conf, # Chỉ dùng điểm tự tin của OCR
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
        # 1. Tiền xử lý: Grayscale + Resize
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Xử lý nâng cao: Khử nhiễu biên và trị bóng râm/chói sáng
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blur)
        
        # 3. Đọc OCR với Allowlist
        ocr_results = reader.readtext(enhanced, detail=1, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
        
        raw_text = ""
        confidences = []
        for bbox, text, prob in ocr_results:
            raw_text += text
            confidences.append(prob)
            
        # 4. Tính điểm tự tin trung bình của chuỗi OCR
        avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        # 5. Hậu xử lý Regex: Định dạng lại theo chuẩn biển số VN
        final_text = self._format_plate_vietnam(raw_text)

        # Đã xóa đoạn code cũ dư thừa ở đây
        return (final_text if final_text else "UNKNOWN"), round(avg_conf, 2)
    
    def _format_plate_vietnam(self, text: str) -> str:
        """
        Business Logic Pro Max: Tự động nhận diện biển 4 số / 5 số 
        và ép kiểu ký tự cực mạnh.
        """
        pure_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(pure_text) < 7 or len(pure_text) > 9: 
            return pure_text
        
        # 1. Xác định độ dài phần đuôi (Biển mới 5 số, biển cũ 4 số)
        # Nếu tổng 9 ký tự (VD: 63B319563) -> Đuôi chắc chắn 5 số
        if len(pure_text) >= 8:
            tail_len = 5 if len(pure_text) == 9 else 4 
            # (Riêng ô tô 51F12345 dài 8 ký tự thì đuôi 5 số, nắn logic tạm ở đây là 4/5)
            # Tối ưu nhất cho xe máy 9 ký tự: đuôi 5.
            if len(pure_text) == 8 and pure_text[2].isalpha() and not pure_text[3].isalpha():
                tail_len = 5 # Xử lý cho biển ô tô 51F 12345
        else:
            tail_len = 4
            
        prefix = pure_text[:-tail_len] # VD: 63B3
        tail = pure_text[-tail_len:]   # VD: J9563
        
        # 2. Nắn gân phần Đầu (2 ký tự đầu BẮT BUỘC là số)
        h12 = prefix[:2].translate(str.maketrans('OIZBSG', '012856'))
        # Ký tự thứ 3 luôn là Chữ cái (Seri) -> Diệt chữ số
        char_3 = prefix[2].translate(str.maketrans('01284', 'OIZBA'))
        # Ký tự thứ 4 để TỰ DO (Có thể là số như B3, F1 hoặc chữ như AA của xe < 50cc)
        char_4 = prefix[3]
        formatted_prefix = f"{h12}-{char_3}{char_4}"
       # 3. NẮN GÂN PHẦN ĐUÔI (TAIL)
        # 100% phải là SỐ -> Diệt sạch mọi ảo giác sinh ra chữ cái
        tail = tail.translate(str.maketrans('OILZSBQAJGTD', '011258041610'))
        
        # 4. GẮN LẠI THÀNH CHUỖI CHUẨN
        if len(tail) == 5:
            # Nếu đuôi 5 số thì nhét dấu chấm vào giữa
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
