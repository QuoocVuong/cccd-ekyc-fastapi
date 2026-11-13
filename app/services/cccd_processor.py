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
    BasicOCRLine,
    BasicOCRResponse,
    CCCDVerificationResponse,
    FaceMatchResponse,
    OCRInfo,
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

    def process(self, image_bytes: bytes, strict_qr: bool = True) -> CCCDVerificationResponse:
        """
        Thực thi toàn bộ quy trình eKYC CCCD Việt Nam tuần tự theo luồng Fail-Fast:
        1. Decode ảnh, resize 640px bắt 4 góc thẻ, validate hình học (hỗ trợ ảnh chụp xa tự cắt khung).
        2. Nắn phẳng thẻ về chuẩn 1000 x 630.
        3. Kiểm tra chất lượng (mờ, sáng/tối, lóa flash, moiré màn hình).
        4. Song song Luồng A (QR) & Luồng B (Chân dung/Embedding), hỗ trợ tự xoay 180°.
        5. OCR Số CCCD và Họ tên, đối chiếu chéo với dữ liệu QR (hoặc OCR mặt trước với Thẻ Căn cước 2024).
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
        # Validate các tiêu chuẩn hình học (thiếu góc, sát mép, tỷ lệ diện tích, aspect ratio)
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
        quality_metrics = self.quality_checker.validate_card_quality(warped)
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

        # Kiểm tra kết quả Luồng B (Ảnh chân dung & Vector)
        if face_result is None:
            raise FaceDetectionError(
                "Không nhận diện được khuôn mặt chân dung hợp lệ trên thẻ "
                f"(yêu cầu độ tin cậy tối thiểu {settings.FACE_DET_SCORE_THRESHOLD * 100:.0f}%)."
            )
        portrait_base64, face_embedding = face_result

        # Kiểm tra kết quả Luồng A (QR)
        qr_info: Optional[QRInfo] = None
        if raw_qr_text is not None:
            qr_info = self.qr_reader.parse_and_validate(raw_qr_text)
        elif strict_qr:
            raise QRCodeError(
                "Không thể phát hiện hoặc giải mã mã QR trên thẻ CCCD. "
                "Nếu đây là Thẻ Căn cước mẫu 2024 (QR ở mặt sau), hãy bật chế độ hỗ trợ 2 loại thẻ (strict_qr=false)."
            )

        qr_and_face_parallel_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 5. OCR & Đối chiếu chéo (hoặc OCR mặt trước Thẻ Căn cước 2024)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if qr_info is not None:
            card_type = "CCCD_2021"
            ocr_info = self.ocr_verifier.verify_and_cross_check(warped, qr_info, executor=self.executor)
        else:
            card_type = "CAN_CUOC_2024"
            full_ocr = self.ocr_verifier.extract_full_card_ocr(warped)
            cccd_num = full_ocr.get("cccd_number") or ""
            # Tìm dòng họ tên in hoa dài nhất sau dòng số CCCD
            raw_lines = full_ocr.get("raw_lines", [])
            name_guess = ""
            for idx, line in enumerate(raw_lines):
                if cccd_num and cccd_num in line["text"] and idx + 1 < len(raw_lines):
                    name_guess = raw_lines[idx + 1]["text"]
                    break
            avg_conf = (
                sum(l["confidence"] for l in raw_lines) / len(raw_lines)
                if raw_lines else 0.0
            )
            ocr_info = OCRInfo(
                cccd_number=cccd_num,
                full_name=name_guess,
                cccd_match=bool(cccd_num),
                name_match=bool(name_guess),
                confidence_score=round(avg_conf, 4),
            )
        ocr_cross_verify_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CCCDVerificationResponse(
            status="success",
            is_valid=True,
            card_type=card_type,
            rotated_180=rotated_180,
            qr_info=qr_info,
            ocr_info=ocr_info,
            portrait_base64=portrait_base64,
            face_embedding=face_embedding,
            quality_metrics=quality_metrics,
            processing_time_ms=ProcessingTimeMS(
                total_ms=round(total_ms, 2),
                corner_detection_ms=round(corner_detection_ms, 2),
                perspective_warp_ms=round(perspective_warp_ms, 2),
                quality_check_ms=round(quality_check_ms, 2),
                qr_and_face_parallel_ms=round(qr_and_face_parallel_ms, 2),
                ocr_cross_verify_ms=round(ocr_cross_verify_ms, 2),
            ),
        )

    def match_face(
        self,
        selfie_bytes: bytes,
        card_embedding: list[float],
    ) -> FaceMatchResponse:
        """
        So khớp khuôn mặt 1:1 giữa ảnh Selfie và vector embedding 512 chiều từ thẻ CCCD.
        """
        t0 = time.perf_counter()
        np_arr = np.frombuffer(selfie_bytes, np.uint8)
        selfie_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if selfie_img is None:
            raise GeometricValidationError("Không thể đọc dữ liệu ảnh Selfie.")

        selfie_emb, det_score = self.face_extractor.extract_selfie_embedding(selfie_img)
        similarity = self.face_extractor.compute_cosine_similarity(card_embedding, selfie_emb)
        is_match = similarity >= settings.FACE_MATCH_SIMILARITY_THRESHOLD

        # Quy đổi phần trăm trực quan: [-1..1] -> [0..100], với threshold 0.40 ~ 70%
        similarity_percent = max(0.0, min(100.0, (similarity + 0.2) / 1.2 * 100.0)) if similarity > 0 else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return FaceMatchResponse(
            status="success",
            is_match=is_match,
            similarity=round(similarity, 4),
            similarity_percent=round(similarity_percent, 1),
            threshold=settings.FACE_MATCH_SIMILARITY_THRESHOLD,
            selfie_det_score=round(det_score, 4),
            processing_time_ms=round(elapsed_ms, 2),
        )

    def ocr_basic(self, image_bytes: bytes) -> BasicOCRResponse:
        """
        OCR cơ bản toàn bộ mặt thẻ CCCD:
        Bắt 4 góc, nắn phẳng 1000x630 và chạy RapidOCR (use_det=True) đọc toàn bộ chữ trên thẻ.
        """
        t0 = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise GeometricValidationError("Không thể đọc dữ liệu hình ảnh.")

        orig_h, orig_w = img.shape[:2]
        corners = self.detector.detect(img)
        if corners is not None:
            try:
                validated = validate_geometry(corners, orig_w, orig_h)
                warped = warp_card(img, validated)
            except GeometricValidationError:
                warped = img
        else:
            warped = img

        ocr_data = self.ocr_verifier.extract_full_card_ocr(warped)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BasicOCRResponse(
            status="success",
            cccd_number=ocr_data["cccd_number"],
            dob=ocr_data["dob"],
            gender=ocr_data["gender"],
            expiry_date=ocr_data["expiry_date"],
            raw_lines=[BasicOCRLine(**line) for line in ocr_data["raw_lines"]],
            processing_time_ms=round(elapsed_ms, 2),
        )
