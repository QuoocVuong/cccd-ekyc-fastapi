# Sử dụng Python 3.11 Slim làm base image
FROM python:3.11-slim

# Thiết lập biến môi trường tối ưu hiệu năng CPU và tắt buffer log
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV, TurboJPEG và build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libturbojpeg0-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn dự án
COPY app /app/app
COPY scripts /app/scripts

# Tải sẵn toàn bộ model trong lúc build Docker Image để container khởi động tức thì (Zero cold start)
RUN python scripts/download_models.py

# Expose cổng dịch vụ
EXPOSE 8000

# Khởi chạy FastAPI với Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
