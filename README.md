# Activate venv trước
.venv\Scripts\activate

# Cài torch với CUDA trước
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Rồi cài phần còn lại
pip install -r requirements.txt

# Chạy server
uvicorn app.main:app --reload

Sau khi xong thì chạy server và test tại http://localhost:8000/docs 
