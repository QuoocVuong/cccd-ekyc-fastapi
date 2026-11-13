#!/usr/bin/env python3
"""
Script tự động tải và khởi tạo sẵn toàn bộ model cần thiết:
1. 4 file WeChatQRCode: detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel
2. InsightFace buffalo_sc: det_500m.onnx, w600k_mbf.onnx
3. DocAligner ONNX: docaligner-lcnet100.onnx
4. RapidOCR Recognition model

Chạy script này trong quá trình docker build để container có đầy đủ model offline và khởi động tức thì.
"""

import os
import shutil
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
WECHAT_DIR = MODELS_DIR / "wechat_qrcode"

WECHAT_FILES = {
    "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
    "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
    "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
    "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel",
}

DOCALIGNER_URL = (
    "https://huggingface.co/kittypdf/DocAligner-ONNX/resolve/main/docaligner-lcnet100.onnx"
)


def download_file(url: str, dest_path: Path) -> None:
    """Tải file từ URL với stream và lưu vào đường dẫn dest_path."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[ĐÃ CÓ] {dest_path.name} ({dest_path.stat().st_size:,} bytes), bỏ qua.")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    print(f"[ĐANG TẢI] {dest_path.name} từ {url} ...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\r  Tiến độ: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)

    print()
    shutil.move(temp_path, dest_path)
    print(f"[HOÀN THÀNH] Đã lưu {dest_path.name} ({dest_path.stat().st_size:,} bytes).")


def setup_wechat_qr() -> None:
    """Tải 4 file model của WeChatQRCode."""
    print("\n=== 1. TẢI MODEL WECHAT QR CODE ===")
    WECHAT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in WECHAT_FILES.items():
        dest = WECHAT_DIR / filename
        download_file(url, dest)


def setup_docaligner() -> None:
    """Tải model DocAligner ONNX."""
    print("\n=== 2. TẢI MODEL DOCALIGNER ONNX ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = MODELS_DIR / "docaligner-lcnet100.onnx"
    try:
        download_file(DOCALIGNER_URL, doc_path)
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tải DocAligner ONNX: {e}. Hệ thống sẽ sử dụng ContourCardDetector cực nhanh.")


def setup_insightface() -> None:
    """Tải và khởi tạo model InsightFace buffalo_sc."""
    print("\n=== 3. KHỞI TẠO INSIGHTFACE (BUFFALO_SC) ===")
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_sc",
            root=str(BASE_DIR),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        print("[HOÀN THÀNH] Model InsightFace buffalo_sc đã sẵn sàng!")
    except Exception as e:
        print(f"[LỖI] Khởi tạo InsightFace thất bại: {e}")
        sys.exit(1)


def setup_rapidocr() -> None:
    """Khởi tạo và cache model RapidOCR."""
    print("\n=== 4. KHỞI TẠO RAPIDOCR ===")
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False)
        # Chạy inference thử 1 ảnh trắng nhỏ để đảm bảo model được cache vào bộ nhớ
        dummy = np.zeros((32, 100, 3), dtype=np.uint8)
        engine(dummy, use_det=False, use_cls=False, use_rec=True)
        print("[HOÀN THÀNH] Model RapidOCR đã được nạp và kiểm tra thành công!")
    except Exception as e:
        print(f"[LỖI] Khởi tạo RapidOCR thất bại: {e}")
        sys.exit(1)


ASTERNET_FILES = {
    "asternet.onnx": "https://huggingface.co/asterxlabs/asternet-vietnamese-ocr/resolve/main/asternet.onnx",
    "asternet_vocab.json": "https://huggingface.co/asterxlabs/asternet-vietnamese-ocr/resolve/main/vocab.json",
}


def setup_asternet() -> None:
    """Tải model AsterNet Vietnamese OCR ONNX (227 ký tự tiếng Việt có dấu chuẩn NFC)."""
    print("\n=== 5. TẢI MODEL ASTERNET VIETNAMESE OCR ONNX ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in ASTERNET_FILES.items():
        download_file(url, MODELS_DIR / filename)


def main() -> None:
    print("==========================================================")
    print(" BẮT ĐẦU TẢI VÀ CHUẨN BỊ MODEL CHO CCCD EKYC BACKEND ")
    print("==========================================================")
    setup_wechat_qr()
    setup_docaligner()
    setup_insightface()
    setup_rapidocr()
    setup_asternet()
    print("\n==========================================================")
    print(" TẤT CẢ MODEL ĐÃ ĐƯỢC CHUẨN BỊ SẴN SÀNG! ")
    print("==========================================================")


if __name__ == "__main__":
    main()
