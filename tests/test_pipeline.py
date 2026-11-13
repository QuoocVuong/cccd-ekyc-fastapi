import io
import time
from typing import Tuple

import cv2
import numpy as np
import pytest
import qrcode
from fastapi.testclient import TestClient
from skimage import data

from app.main import app

client = TestClient(app)


def create_synthetic_cccd_image(
    cccd_number: str = "001099012345",
    full_name: str = "NGUYEN VAN A",
    mismatch_number_on_card: str = None,
    rotate_180: bool = False,
    add_blur: bool = False,
    add_glare_qr: bool = False,
    near_edge: bool = False,
    too_small: bool = False,
) -> bytes:
    """
    Tạo ảnh thẻ CCCD nhân tạo hợp lệ hoặc mô phỏng các trường hợp lỗi để kiểm thử.
    """
    card_bg_color = (215, 230, 225)  # Màu nhựa thẻ CCCD chuẩn tự nhiên (V < 240)
    card = np.full((630, 1000, 3), card_bg_color, dtype=np.uint8)

    # 1. Mã QR tại (10..290, 680..990)
    qr_data = f"{cccd_number}||{full_name}|15081995|Nam|Ha Noi|20112021"
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_matrix = np.array(qr.get_matrix(), dtype=bool)

    qr_h, qr_w = qr_matrix.shape
    scale = 4
    qr_img = np.full((qr_h * scale, qr_w * scale, 3), card_bg_color, dtype=np.uint8)
    for r in range(qr_h):
        for c in range(qr_w):
            if qr_matrix[r, c]:
                qr_img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = (15, 15, 15)

    qrh, qrw = qr_img.shape[:2]
    card[40 : 40 + qrh, 740 : 740 + qrw] = qr_img

    # 2. Ảnh chân dung tại (140..580, 30..370)
    astronaut = data.astronaut()
    face = astronaut[30:260, 160:340]
    face_resized = cv2.resize(face, (260, 340))
    face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
    card[180 : 180 + 340, 60 : 60 + 260] = face_bgr

    # 3. Số CCCD tại (245..315, 390..760)
    card_number_text = mismatch_number_on_card or cccd_number
    cv2.putText(card, card_number_text, (400, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3)

    # 4. Họ và tên tại (320..395, 385..920)
    cv2.putText(card, full_name, (400, 365), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (15, 15, 15), 3)

    # Xử lý xoay 180 độ nếu test case yêu cầu
    if rotate_180:
        card = cv2.rotate(card, cv2.ROTATE_180)

    # Xử lý lóa đèn flash trên vùng QR
    if add_glare_qr:
        # V > 245 và S < 30
        card[20:150, 700:850] = (255, 255, 255)

    # Đặt thẻ lên ảnh nền tổng thể
    if too_small:
        # Thẻ chiếm < 22% diện tích
        full_img = np.full((1500, 2000, 3), 40, dtype=np.uint8)
        small_card = cv2.resize(card, (500, 315))
        full_img[500 : 500 + 315, 750 : 750 + 500] = small_card
    elif near_edge:
        # Đặt thẻ sát mép trái (x < 8px)
        full_img = np.full((960, 1280, 3), 40, dtype=np.uint8)
        full_img[165 : 165 + 630, 2 : 2 + 1000] = card
    else:
        # Chuẩn: Thẻ ở giữa, diện tích ~51%
        full_img = np.full((960, 1280, 3), 40, dtype=np.uint8)
        full_img[165 : 165 + 630, 140 : 140 + 1000] = card

    # Xử lý ảnh mờ nhòe nét
    if add_blur:
        full_img = cv2.GaussianBlur(full_img, (35, 35), 0)

    success, encoded = cv2.imencode(".jpg", full_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes()


def test_health_check():
    """Kiểm tra endpoint /health trả về 200 và các model đã nạp."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["processor_initialized"] is True
    assert data["models_loaded"]["wechat_qr"] is True
    assert data["models_loaded"]["insightface"] is True
    assert data["models_loaded"]["rapid_ocr"] is True


def test_root_endpoint():
    """Kiểm tra endpoint / trả về giao diện Web UI và /api/info trả về JSON."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "CCCD Vietnam eKYC Console" in response.text

    info_resp = client.get("/api/info")
    assert info_resp.status_code == 200
    assert "verify_endpoint" in info_resp.json()


def test_verify_valid_cccd():
    """Kiểm tra pipeline xác thực CCCD thành công với ảnh hợp lệ."""
    img_bytes = create_synthetic_cccd_image()
    files = {"file": ("test_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 200
    res = response.json()

    assert res["status"] == "success"
    assert res["is_valid"] is True
    assert res["rotated_180"] is False

    # Kiểm tra dữ liệu QR
    qr = res["qr_info"]
    assert qr["cccd_number"] == "001099012345"
    assert qr["full_name"] == "NGUYEN VAN A"
    assert qr["dob"] == "15081995"
    assert qr["gender"] == "Nam"
    assert qr["address"] == "Ha Noi"
    assert qr["issue_date"] == "20112021"

    # Kiểm tra OCR và đối chiếu chéo
    ocr = res["ocr_info"]
    assert ocr["cccd_number"] == "001099012345"
    assert ocr["cccd_match"] is True
    assert ocr["name_match"] is True

    # Kiểm tra ảnh chân dung & embedding
    assert res["portrait_base64"].startswith("data:image/jpeg;base64,")
    assert len(res["face_embedding"]) == 512

    # Kiểm tra thống kê thời gian
    times = res["processing_time_ms"]
    assert times["total_ms"] > 0
    assert times["corner_detection_ms"] > 0
    assert times["perspective_warp_ms"] > 0
    assert times["quality_check_ms"] > 0
    assert times["qr_and_face_parallel_ms"] > 0
    assert times["ocr_cross_verify_ms"] > 0


def test_auto_180_rotation():
    """Kiểm tra tính năng tự động xoay thẻ 180 độ khi ảnh bị cắm ngược."""
    img_bytes = create_synthetic_cccd_image(rotate_180=True)
    files = {"file": ("rotated_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["rotated_180"] is True
    assert res["qr_info"]["cccd_number"] == "001099012345"


def test_fail_fast_blurred_image():
    """Kiểm tra Fail-Fast khi ảnh bị mờ nét (Laplacian var < 85.0)."""
    img_bytes = create_synthetic_cccd_image(add_blur=True)
    files = {"file": ("blurred_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "IMAGE_QUALITY_ERROR"
    assert "mờ" in data["message"].lower() or "nhòe" in data["message"].lower()


def test_fail_fast_glare_flash():
    """Kiểm tra Fail-Fast khi ảnh bị lóa đèn flash trên vùng QR (> 2%)."""
    img_bytes = create_synthetic_cccd_image(add_glare_qr=True)
    files = {"file": ("glare_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "IMAGE_QUALITY_ERROR"
    assert "lóa" in data["message"].lower()


def test_fail_fast_edge_proximity():
    """Kiểm tra Fail-Fast khi góc thẻ nằm sát mép ảnh dưới 8px."""
    img_bytes = create_synthetic_cccd_image(near_edge=True)
    files = {"file": ("near_edge_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "GEOMETRIC_VALIDATION_ERROR"
    assert "sát mép" in data["message"].lower()


def test_fail_fast_area_too_small():
    """Kiểm tra Fail-Fast khi thẻ quá xa camera (< 22% diện tích)."""
    img_bytes = create_synthetic_cccd_image(too_small=True)
    files = {"file": ("small_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "GEOMETRIC_VALIDATION_ERROR"
    assert "quá xa" in data["message"].lower()


def test_fail_fast_cross_verification_mismatch():
    """Kiểm tra Fail-Fast khi Số CCCD trên mặt thẻ khác số trong QR (phát hiện sửa ảnh)."""
    img_bytes = create_synthetic_cccd_image(mismatch_number_on_card="001099999999")
    files = {"file": ("forged_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "CROSS_VERIFICATION_MISMATCH"
    assert "không trùng khớp" in data["message"].lower()


def test_benchmark_latency():
    """Đo lường benchmark hiệu năng xử lý liên tục qua API client."""
    img_bytes = create_synthetic_cccd_image()
    files_fn = lambda: {"file": ("bench.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    # Warmup
    client.post("/api/v1/cccd/verify", files=files_fn())
    client.post("/api/v1/cccd/verify", files=files_fn())

    latencies = []
    print("\n" + "=" * 60)
    print(" BENCHMARK HIỆU NĂNG PIPELINE EKYC CCCD (10 REQUESTS) ")
    print("=" * 60)

    for i in range(10):
        t0 = time.perf_counter()
        resp = client.post("/api/v1/cccd/verify", files=files_fn())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies.append(elapsed_ms)
        breakdown = resp.json()["processing_time_ms"]
        print(f" Request {i+1:2d}: {elapsed_ms:6.2f} ms | Internal Total: {breakdown['total_ms']:6.2f} ms "
              f"(QR+Face: {breakdown['qr_and_face_parallel_ms']:5.1f}ms, OCR: {breakdown['ocr_cross_verify_ms']:5.1f}ms)")

    avg_latency = sum(latencies) / len(latencies)
    print("-" * 60)
    print(f" Tốc độ trung bình (End-to-End API): {avg_latency:.2f} ms / request")
    print("=" * 60)


def test_face_match_and_ocr_basic():
    """Kiểm tra endpoint Face Matching 1:1 và OCR cơ bản toàn thẻ."""
    img_bytes = create_synthetic_cccd_image()

    # 1. Test OCR cơ bản toàn thẻ
    ocr_resp = client.post(
        "/api/v1/cccd/ocr-basic",
        files={"file": ("card.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert ocr_resp.status_code == 200
    ocr_data = ocr_resp.json()
    assert ocr_data["status"] == "success"
    assert ocr_data["cccd_number"] == "001099012345"
    assert len(ocr_data["raw_lines"]) >= 2

    # 2. Test Face Matching 1:1 (dùng chính ảnh khuôn mặt astronaut làm selfie)
    astronaut = data.astronaut()
    face_bgr = cv2.cvtColor(astronaut[30:260, 160:340], cv2.COLOR_RGB2BGR)
    _, selfie_buf = cv2.imencode(".jpg", face_bgr)

    match_resp = client.post(
        "/api/v1/cccd/face-match",
        files={
            "selfie_file": ("selfie.jpg", io.BytesIO(selfie_buf.tobytes()), "image/jpeg"),
            "cccd_file": ("card.jpg", io.BytesIO(img_bytes), "image/jpeg"),
        },
    )
    assert match_resp.status_code == 200
    match_data = match_resp.json()
    assert match_data["status"] == "success"
    assert match_data["is_match"] is True
    assert match_data["similarity"] >= 0.70
