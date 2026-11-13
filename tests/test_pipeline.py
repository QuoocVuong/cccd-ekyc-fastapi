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
        # Thẻ chiếm < 4% diện tích (500*315 / (2500*2000) = 3.15% < 5%)
        full_img = np.full((2000, 2500, 3), 40, dtype=np.uint8)
        small_card = cv2.resize(card, (500, 315))
        full_img[800 : 800 + 315, 1000 : 1000 + 500] = small_card
    elif near_edge:
        # Đặt thẻ sát mép trái (x < 4px)
        full_img = np.full((960, 1280, 3), 40, dtype=np.uint8)
        full_img[165 : 165 + 630, 1 : 1 + 1000] = card
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

    response = client.post("/api/v1/cccd/verify?allow_screen=false", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "GEOMETRIC_VALIDATION_ERROR"
    assert "sát mép" in data["message"].lower()


def test_fail_fast_area_too_small():
    """Kiểm tra Fail-Fast khi thẻ quá xa camera (< 22% diện tích)."""
    img_bytes = create_synthetic_cccd_image(too_small=True)
    files = {"file": ("small_cccd.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/cccd/verify?allow_screen=false", files=files)
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


def test_vneid_screenshot_and_vietnamese_diacritics():
    """Kiểm tra nhận diện ảnh chụp màn hình VNeID (dọc) và OCR tiếng Việt có dấu."""
    from PIL import Image, ImageDraw, ImageFont

    # Tạo ảnh màn hình điện thoại dọc 720x1280
    screen = np.full((1280, 720, 3), (242, 245, 248), dtype=np.uint8)

    # Vẽ QR code góc trên phải
    qr_data = "002099012345||LÃNH QUỐC VƯƠNG|15081995|Nam|Cao Lộc, Lạng Sơn|20112021"
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_matrix = np.array(qr.get_matrix(), dtype=bool)
    qrh, qrw = qr_matrix.shape
    scale = 4
    qr_img = np.full((qrh * scale, qrw * scale, 3), 245, dtype=np.uint8)
    for r in range(qrh):
        for c in range(qrw):
            if qr_matrix[r, c]:
                qr_img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = (15, 15, 15)
    qh, qw = qr_img.shape[:2]
    screen[120 : 120 + qh, 480 : 480 + qw] = qr_img

    # Vẽ chân dung bên trái
    astronaut = data.astronaut()
    face = cv2.resize(astronaut[30:260, 160:340], (200, 260))
    screen[120 : 120 + 260, 40 : 40 + 200] = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    # Vẽ chữ tiếng Việt có dấu bằng Pillow
    pil_img = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font_lg = ImageFont.truetype(font_path, 34)
        font_md = ImageFont.truetype(font_path, 28)
    except Exception:
        font_lg = ImageFont.load_default()
        font_md = ImageFont.load_default()

    draw.text((60, 440), "Số ĐDCN: 002099012345", fill=(15, 15, 15), font=font_lg)
    draw.text((60, 500), "LÃNH QUỐC VƯƠNG", fill=(15, 15, 15), font=font_lg)
    draw.text((60, 560), "Ngày sinh: 15/08/1995", fill=(30, 30, 30), font=font_md)
    draw.text((60, 610), "Giới tính: Nam", fill=(30, 30, 30), font=font_md)

    screen_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", screen_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    response = client.post(
        "/api/v1/cccd/verify?strict_qr=false&allow_screen=true",
        files={"file": ("vneid_screen.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["card_type"] == "VNEID_SCREEN"
    assert res["qr_info"]["cccd_number"] == "002099012345"
    assert res["ocr_info"]["cccd_number"] == "002099012345"
    assert "VƯƠNG" in res["ocr_info"]["full_name"] or "VUONG" in res["ocr_info"]["full_name"]
    assert res["portrait_base64"].startswith("data:image/jpeg;base64,")
    assert len(res["face_embedding"]) == 512


def test_fail_fast_clipped_or_occluded_corner():
    """Kiểm tra Fail-Fast khi thẻ bị cắt góc / che góc hoặc che thông tin ngay cả khi allow_screen=true."""
    img_bytes = create_synthetic_cccd_image()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Che góc trên-trái của thẻ bằng vùng đen (giả lập ngón tay/lẹm góc tại x=140..300, y=165..295)
    img[165:295, 140:300] = (5, 5, 5)
    _, buf = cv2.imencode(".jpg", img)

    response = client.post(
        "/api/v1/cccd/verify?strict_qr=false&allow_screen=true",
        files={"file": ("occluded_corner.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] in ("GEOMETRIC_VALIDATION_ERROR", "IMAGE_QUALITY_ERROR")


def test_vneid_2024_without_qr_screen():
    """Kiểm tra đọc ảnh chụp màn hình VNeID mẫu mới (Căn cước 2024 không có QR mặt trước)."""
    from PIL import Image, ImageDraw, ImageFont

    screen = np.full((1280, 720, 3), (240, 244, 248), dtype=np.uint8)
    astronaut = data.astronaut()
    face = cv2.resize(astronaut[30:260, 160:340], (200, 260))
    screen[120 : 120 + 260, 40 : 40 + 200] = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font_lg = ImageFont.truetype(font_path, 32)
        font_md = ImageFont.truetype(font_path, 26)
    except Exception:
        font_lg = ImageFont.load_default()
        font_md = ImageFont.load_default()

    draw.text((270, 130), "CĂN CƯỚC ĐIỆN TỬ", fill=(160, 25, 25), font=font_lg)
    draw.text((270, 200), "Số định danh cá nhân:", fill=(70, 70, 70), font=font_md)
    draw.text((270, 240), "002099887766", fill=(15, 15, 15), font=font_lg)
    draw.text((60, 430), "Họ, chữ đệm và tên khai sinh:", fill=(70, 70, 70), font=font_md)
    draw.text((60, 470), "NGUYỄN THỊ ÁNH TUYẾT", fill=(15, 15, 15), font=font_lg)
    draw.text((60, 540), "Ngày, tháng, năm sinh: 20/10/1998", fill=(30, 30, 30), font=font_md)
    draw.text((60, 590), "Giới tính: Nữ", fill=(30, 30, 30), font=font_md)

    screen_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", screen_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    response = client.post(
        "/api/v1/cccd/verify?strict_qr=false&allow_screen=true",
        files={"file": ("vneid_2024.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["card_type"] == "VNEID_SCREEN"
    assert res["qr_info"] is None
    assert res["ocr_info"]["cccd_number"] == "002099887766"
    assert "TUYẾT" in res["ocr_info"]["full_name"] or "TUYET" in res["ocr_info"]["full_name"]
    assert res["portrait_base64"].startswith("data:image/jpeg;base64,")


def test_back_side_cccd_mrz_and_side_mismatch():
    """Kiểm tra tự động nhận diện MẶT SAU (MRZ + Đặc điểm) và bắt lỗi SIDE_MISMATCH."""
    from PIL import Image, ImageDraw, ImageFont

    card = np.full((630, 1000, 3), (215, 228, 224), dtype=np.uint8)
    pil_card = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_card)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    try:
        font_mrz = ImageFont.truetype(font_path, 30)
        font_txt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font_mrz = ImageFont.load_default()
        font_txt = ImageFont.load_default()

    draw.text((50, 80), "Đặc điểm nhận dạng: Nốt ruồi", fill=(20, 20, 20), font=font_txt)
    draw.text((50, 140), "Ngày cấp: 20/11/2021", fill=(20, 20, 20), font=font_txt)
    draw.text((50, 190), "BỘ CÔNG AN", fill=(20, 20, 20), font=font_txt)

    # 3 dòng MRZ chuẩn TD1
    draw.text((50, 440), "IDVNM0990123456001099012345<<9", fill=(10, 10, 10), font=font_mrz)
    draw.text((50, 490), "9508152M3108156VNM<<<<<<<<<<<0", fill=(10, 10, 10), font=font_mrz)
    draw.text((50, 540), "LANH<<QUOC<VUONG<<<<<<<<<<<<<<", fill=(10, 10, 10), font=font_mrz)

    card_bgr = cv2.cvtColor(np.array(pil_card), cv2.COLOR_RGB2BGR)
    full_img = np.full((960, 1280, 3), 40, dtype=np.uint8)
    full_img[165 : 165 + 630, 140 : 140 + 1000] = card_bgr
    _, buf = cv2.imencode(".jpg", full_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 1. Auto-detect BACK
    resp = client.post(
        "/api/v1/cccd/verify?strict_qr=false&expected_side=AUTO",
        files={"file": ("back_cccd.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["card_side"] == "BACK"
    assert res["request_id"]
    assert res["timestamp"]
    assert res["mrz_info"] is not None
    assert res["mrz_info"]["document_number"] == "001099012345"
    assert res["back_info"] is not None

    # 2. Expected FRONT nhưng gửi BACK -> SIDE_MISMATCH
    mismatch_resp = client.post(
        "/api/v1/cccd/verify?strict_qr=false&expected_side=FRONT",
        files={"file": ("back_cccd.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert mismatch_resp.status_code == 422
    assert mismatch_resp.json()["error_code"] == "SIDE_MISMATCH"


def test_vneid_dual_side_screen():
    """Kiểm tra nhận diện ảnh chụp màn hình VNeID loại 2 mặt (Mặt trước + Mặt sau trên cùng ảnh)."""
    from PIL import Image, ImageDraw, ImageFont

    screen = np.full((1500, 720, 3), (240, 244, 248), dtype=np.uint8)
    astronaut = data.astronaut()
    face = cv2.resize(astronaut[30:260, 160:340], (200, 260))
    screen[100 : 100 + 260, 40 : 40 + 200] = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_mrz = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)
    except Exception:
        font_lg = font_md = font_mrz = ImageFont.load_default()

    # Nửa trên: Mặt trước VNeID
    draw.text((260, 140), "Số định danh cá nhân:", fill=(70, 70, 70), font=font_md)
    draw.text((260, 180), "002099012345", fill=(15, 15, 15), font=font_lg)
    draw.text((40, 400), "Họ và tên: LÃNH QUỐC VƯƠNG", fill=(15, 15, 15), font=font_lg)

    # Nửa dưới: Mặt sau VNeID
    draw.text((40, 760), "Đặc điểm nhận dạng: Sẹo chấm", fill=(30, 30, 30), font=font_md)
    draw.text((40, 810), "Ngày cấp: 20/11/2021 - BỘ CÔNG AN", fill=(30, 30, 30), font=font_md)
    draw.text((40, 920), "IDVNM0990123456002099012345<<9", fill=(10, 10, 10), font=font_mrz)
    draw.text((40, 960), "9508152M3108156VNM<<<<<<<<<<<0", fill=(10, 10, 10), font=font_mrz)
    draw.text((40, 1000), "LANH<<QUOC<VUONG<<<<<<<<<<<<<<", fill=(10, 10, 10), font=font_mrz)

    screen_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", screen_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    resp = client.post(
        "/api/v1/cccd/verify?strict_qr=false&allow_screen=true",
        files={"file": ("vneid_dual.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["card_type"] == "VNEID_SCREEN"
    assert res["card_side"] == "BOTH"
    assert res["vneid_layout"] == "DUAL_SIDE"
    assert res["mrz_info"] is not None
    assert res["back_info"] is not None
    assert res["portrait_base64"].startswith("data:image/jpeg;base64,")



