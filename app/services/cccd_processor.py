import concurrent.futures
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    FaceDetectionError,
    GeometricValidationError,
    QRCodeError,
)
from app.schemas.cccd import (
    CCCDVerificationResponse,
    ProcessingTimeMS,
    QRInfo,
)
from app.services.card_detector import (
    HybridCardDetector,
    validate_geometry,
    warp_card,
)
from app.services.face_extractor import FaceExtractor
from app.services.ocr_verifier import OCRVerifier
from app.services.qr_reader import QRReader
from app.services.quality_checker import QualityChecker

logger = logging.getLogger(__name__)


class CCCDProcessor:
    """
    Trình điều phối toàn diện cho pipeline xác thực và trích xuất dữ liệu thẻ CCCD Việt Nam (eKYC).
    Tối ưu hóa hiệu năng cực cao (mục tiêu < 60ms trên CPU) với mô hình Fail-Fast và ThreadPoolExecutor.
    Được khởi tạo theo dạng Singleton một lần duy nhất trong lifespan của FastAPI.
    """

    def __init__(self):
        logger.info("Đang khởi tạo các module của CCCDProcessor Singleton...")
        t0 = time.perf_counter()

        # 1. Bộ bắt góc thẻ (Hybrid: Fast Contour + ONNX DocAligner fallback)
        self.detector = HybridCardDetector(
            onnx_model_path=settings.DOCALIGNER_MODEL_PATH if settings.USE_ONNX_CORNER_DETECTOR else None
        )

        # 2. Bộ kiểm tra chất lượng & Chống giả mạo (Anti-Spoofing)
        self.quality_checker = QualityChecker()

        # 3. Bộ đọc mã QR (WeChatQRCode + zxing-cpp fallback)
        self.qr_reader = QRReader(models_dir=settings.WECHAT_QR_DIR)

        # 4. Bộ cắt ảnh chân dung & trích xuất vector (InsightFace buffalo_sc CPU)
        self.face_extractor = FaceExtractor(
            model_name=settings.INSIGHTFACE_NAME,
            root_dir=settings.BASE_DIR
        )

        # 5. Bộ OCR đối chiếu chéo (RapidOCR recognition only)
        self.ocr_verifier = OCRVerifier()

        # 6. ThreadPoolExecutor cho xử lý song song các vùng ROI
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.THREAD_POOL_WORKERS,
            thread_name_prefix="cccd_worker_"
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Khởi tạo CCCDProcessor hoàn tất trong {elapsed:.1f} ms.")

    def close(self) -> None:
        """Giải phóng tài nguyên thread pool khi tắt ứng dụng."""
        self.executor.shutdown(wait=True)
        logger.info("Đã tắt ThreadPoolExecutor của CCCDProcessor.")

    def _execute_qr_and_face(
        self, warped_card: np.ndarray
    ) -> Tuple[Optional[str], Optional[Tuple[str, list[float]]]]:
        """
        Chạy song song Luồng A (Đọc mã QR) và Luồng B (Cắt chân dung & Vector) qua ThreadPoolExecutor.
        """
        # Cắt ROI cho Luồng A: QR góc trên phải (y: 10..290, x: 680..990)
        qy1, qy2, qx1, qx2 = settings.ROI_QR
        qr_crop = warped_card[qy1:qy2, qx1:qx2]

        # Cắt ROI cho Luồng B: Chân dung bên trái (y: 140..580, x: 30..370)
        fy1, fy2, fx1, fx2 = settings.ROI_PORTRAIT
        face_crop = warped_card[fy1:fy2, fx1:fx2]

        # Đẩy vào ThreadPoolExecutor
        future_qr = self.executor.submit(self.qr_reader.decode_raw, qr_crop)
        future_face = self.executor.submit(
            self.face_extractor.process,
            warped_card,
            face_crop,
            (fy1, fx1)
        )

        raw_qr_text = future_qr.result()
        face_result = future_face.result()

        return raw_qr_text, face_result

    def process(self, image_bytes: bytes) -> CCCDVerificationResponse:
        """
        Thực thi toàn bộ quy trình eKYC CCCD Việt Nam tuần tự theo luồng Fail-Fast:
        1. Decode ảnh, resize 640px bắt 4 góc thẻ, validate hình học.
        2. Nắn phẳng thẻ về chuẩn 1000 x 630.
        3. Kiểm tra chất lượng (mờ, lóa flash, moiré màn hình).
        4. Song song Luồng A (QR) & Luồng B (Chân dung/Embedding), hỗ trợ tự xoay 180°.
        5. OCR Số CCCD và Họ tên, đối chiếu chéo với dữ liệu QR.
        """
        t_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Tiền xử lý & Bắt 4 góc thẻ (Corner Detection & Geometry)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise GeometricValidationError(
                "Không thể đọc dữ liệu hình ảnh. File tải lên không đúng định dạng ảnh hoặc bị hỏng."
            )

        orig_h, orig_w = img.shape[:2]
        corners = self.detector.detect(img)
        # Validate các tiêu chuẩn hình học (thiếu góc, sát mép <8px, tỷ lệ diện tích, aspect ratio)
        validated_corners = validate_geometry(corners, orig_w, orig_h)
        corner_detection_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 2. Nắn phẳng thẻ (Perspective Warp về 1000 x 630)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        warped = warp_card(img, validated_corners)
        perspective_warp_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 3. Kiểm tra chất lượng ảnh & Chống giả mạo (Anti-Spoofing)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        self.quality_checker.validate_card_quality(warped)
        quality_check_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 4. Xử lý song song ROI (Luồng A - QR & Luồng B - Chân dung)
        #    Hỗ trợ tự động xoay ngược 180° nếu thẻ bị cắm ngược
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        rotated_180 = False
        raw_qr_text, face_result = self._execute_qr_and_face(warped)

        # Nếu cả Luồng A và Luồng B đều không tìm thấy dữ liệu, tự động xoay thẻ 180°
        if raw_qr_text is None and face_result is None:
            logger.info("Không tìm thấy QR và Chân dung ở chiều ban đầu. Tiến hành tự xoay thẻ 180 độ...")
            warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
            rot_qr_text, rot_face_result = self._execute_qr_and_face(warped_rot)

            if rot_qr_text is not None or rot_face_result is not None:
                warped = warped_rot
                raw_qr_text = rot_qr_text
                face_result = rot_face_result
                rotated_180 = True

        # Kiểm tra kết quả Luồng A (QR)
        if raw_qr_text is None:
            raise QRCodeError(
                "Không thể phát hiện hoặc giải mã mã QR trên thẻ CCCD. "
                "Vui lòng đảm bảo góc chứa mã QR không bị trầy xước, che khuất hoặc mờ mất nét."
            )
        qr_info = self.qr_reader.parse_and_validate(raw_qr_text)

        # Kiểm tra kết quả Luồng B (Ảnh chân dung & Vector)
        if face_result is None:
            raise FaceDetectionError(
                "Không nhận diện được khuôn mặt chân dung hợp lệ trên thẻ "
                f"(yêu cầu độ tin cậy tối thiểu {settings.FACE_DET_SCORE_THRESHOLD * 100:.0f}%)."
            )
        portrait_base64, face_embedding = face_result
        qr_and_face_parallel_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 5. OCR & Đối chiếu chéo chống sửa ảnh (Cross-Verification)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        ocr_info = self.ocr_verifier.verify_and_cross_check(warped, qr_info, executor=self.executor)
        ocr_cross_verify_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CCCDVerificationResponse(
            status="success",
            is_valid=True,
            rotated_180=rotated_180,
            qr_info=qr_info,
            ocr_info=ocr_info,
            portrait_base64=portrait_base64,
            face_embedding=face_embedding,
            processing_time_ms=ProcessingTimeMS(
                total_ms=round(total_ms, 2),
                corner_detection_ms=round(corner_detection_ms, 2),
                perspective_warp_ms=round(perspective_warp_ms, 2),
                quality_check_ms=round(quality_check_ms, 2),
                qr_and_face_parallel_ms=round(qr_and_face_parallel_ms, 2),
                ocr_cross_verify_ms=round(ocr_cross_verify_ms, 2),
            ),
        )
