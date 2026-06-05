# 🚗 Automatic License Plate Recognition (ALPR) API

A high-performance, containerized Artificial Intelligence pipeline designed to detect and recognize Vietnamese license plates in real-time. This project leverages a hybrid architecture combining Deep Learning (YOLOv8 & EasyOCR) with a Rule-based Expert System (Regex Engine) for maximum accuracy.

## ✨ Key Features
*   **Robust Object Detection:** Fine-tuned YOLOv8 model trained on a custom dataset, resilient to extreme angles and low-light conditions.
*   **Optical Character Recognition:** Utilizes EasyOCR (CRAFT + CRNN) constrained by a strict CTC Allowlist to eliminate background noise.
*   **Smart Post-processing Engine:** A custom 4-layer Regex pipeline that autocorrects OCR hallucinations (e.g., confusing 'X' with 'K', or '0' with 'O') based on Vietnamese license plate formats.
*   **Multi-plate Processing:** Capable of detecting and sorting multiple vehicles in a single frame (Spatial Sorting algorithm).
*   **Asynchronous API:** Built with FastAPI utilizing a Singleton model-loading pattern to eliminate cold starts and reduce latency to ~2 seconds per image.

## 📊 Model Performance
Based on the testing dataset evaluations, the YOLOv8 detection model achieved the following metrics over 50 epochs:
*   **mAP@50:** `99.5%`
*   **mAP@50-95:** `88.0%`
*   **Precision:** `98.5%`
*   **Recall:** `99.0%`

## ⚙️ System Architecture Pipeline
1.  **Input:** Image received via `multipart/form-data` endpoint.
2.  **Detection:** YOLOv8 extracts tight bounding boxes around license plates.
3.  **Pre-processing:** Grayscale conversion (bypassing heavy resizing to preserve pixel integrity).
4.  **Recognition:** EasyOCR extracts raw text strings with a predefined character allowlist.
5.  **Validation:** Regex engine casts data types, filters noise, and formats the output.
6.  **Output:** JSON payload returned to the client.

## 🚀 API Endpoint Reference

### `POST /api/v1/predict`
Accepts an image file and returns detected license plates.

**Request Payload:**
`multipart/form-data` containing `file: <image_bytes>`

**Success Response:**
```json
{
  "filename": "parking_lot_1.jpg",
  "detections": [
    {
      "plate_text": "63-B4 656.24",
      "confidence": 88.45,
      "bbox": {"x1": 150, "y1": 320, "x2": 450, "y2": 510}
    },
    {
      "plate_text": "86-AH 048.10",
      "confidence": 79.12,
      "bbox": {"x1": 600, "y1": 340, "x2": 820, "y2": 520}
    }
  ],
  "total_plates": 2,
  "process_time_ms": 2150.45
}
# Activate venv trước
.venv\Scripts\activate

# Cài torch với CUDA trước
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Rồi cài phần còn lại
pip install -r requirements.txt

# Chạy server
uvicorn app.main:app --reload

Sau khi xong thì chạy server và test tại http://localhost:8000/docs 
