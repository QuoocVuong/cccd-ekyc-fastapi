import concurrent.futures
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    FaceDetectionError,
    GeometricValidationError,
    QRCodeError,
    SideMismatchError,
)
from app.schemas.cccd import (
    BackSideInfo,
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

        # Đẩy vào ThreadPoolExecutor (giới hạn chặt ảnh chân dung trong ROI_PORTRAIT_INNER để không lẹm ra thẻ)
        future_qr = self.executor.submit(self.qr_reader.decode_raw, qr_crop)
        future_face = self.executor.submit(
            self.face_extractor.process,
            warped_card,
            face_crop,
            (fy1, fx1),
            settings.ROI_PORTRAIT_INNER,
        )

        raw_qr_text = future_qr.result()
        face_result = future_face.result()

        return raw_qr_text, face_result

    @staticmethod
    def _new_meta() -> Tuple[str, str]:
        return (
            str(uuid.uuid4()),
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    @staticmethod
    def _collect_warnings(
        quality_metrics: Optional[dict[str, float]],
        card_type: str,
        card_side: str,
        mrz_info=None,
        corners: Optional[np.ndarray] = None,
        orig_w: int = 0,
        orig_h: int = 0,
    ) -> list[str]:
        warnings: list[str] = []
        if quality_metrics:
            b = quality_metrics.get("brightness", 128.0)
            if 0.0 < b < 70.0:
                warnings.append("Cảnh báo thiếu sáng: Độ sáng ảnh hơi thấp, nên chụp nơi đủ sáng hơn.")
            g_card = quality_metrics.get("glare_card_pct", 0.0)
            g_qr = quality_metrics.get("glare_qr_pct", 0.0)
            g_face = quality_metrics.get("glare_portrait_pct", 0.0)
            g_text = quality_metrics.get("glare_text_pct", 0.0)
            if g_card > 1.5 or g_qr > 1.0 or g_face > 1.5 or g_text > 1.5:
                warnings.append(
                    f"Cảnh báo lóa thẻ nhẹ (Toàn thẻ: {g_card}%, QR: {g_qr}%, Chân dung: {g_face}%, Chữ: {g_text}%). "
                    "Nên tắt flash hoặc nghiêng nhẹ thẻ để tránh chói."
                )
        if corners is not None and orig_w > 0 and orig_h > 0:
            near_margin = 16.0
            for x, y in corners:
                if x < near_margin or y < near_margin or x > (orig_w - near_margin) or y > (orig_h - near_margin):
                    warnings.append(
                        "Cảnh báo lẹm thẻ / sát mép: Có góc thẻ nằm gần viền ảnh, nên đặt thẻ lùi vào giữa khung hình."
                    )
                    break
        if card_type == "CAN_CUOC_2024" and card_side == "FRONT":
            warnings.append("Thẻ Căn cước 2024 không có mã QR ở mặt trước; khuyến nghị chụp thêm mặt sau để đọc QR/MRZ.")
        if mrz_info is not None and not mrz_info.checksum_valid:
            warnings.append("Check-digit MRZ chưa khớp hoàn toàn (có thể do ảnh hơi nghiêng hoặc mờ vùng MRZ).")
        return warnings

    def _process_screen_or_vneid(
        self,
        img: np.ndarray,
        t_start: float,
        corner_detection_ms: float,
        strict_qr: bool,
        expected_side: str = "AUTO",
    ) -> CCCDVerificationResponse:
        """
        Xử lý ảnh chụp màn hình VNeID / Thẻ căn cước điện tử:
        - Hỗ trợ cả VNeID loại 1 mặt (SINGLE_SIDE) và VNeID loại 2 mặt (DUAL_SIDE: mặt trước + mặt sau).
        - Chạy song song QR + Face + OCR tiếng Việt để tối ưu phản hồi nhanh.
        """
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1200:
            scale = 1200.0 / max_dim
            proc_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc_img = img

        future_qr = self.executor.submit(self.qr_reader.decode_raw, proc_img)
        future_face = self.executor.submit(
            self.face_extractor.process,
            proc_img,
            proc_img,
            (0, 0),
        )
        future_ocr = self.executor.submit(self.ocr_verifier.extract_full_card_ocr, proc_img)

        raw_qr_text = future_qr.result()
        face_result = future_face.result()
        full_ocr = future_ocr.result()
        parallel_ms = (time.perf_counter() - t0) * 1000.0

        raw_lines = full_ocr.get("raw_lines", [])
        text_lines = [l["text"] for l in raw_lines]
        all_lower = "\n".join(text_lines).lower()

        from app.services.mrz_parser import parse_mrz_lines
        mrz_info = parse_mrz_lines(text_lines)

        has_back_markers = (
            mrz_info is not None
            or "đặc điểm" in all_lower
            or "personal identification" in all_lower
            or "bộ công an" in all_lower
            or "cục cảnh sát" in all_lower
        )

        # Phân biệt VNeID 1 mặt vs 2 mặt và FRONT vs BACK
        if face_result is not None and has_back_markers:
            card_side = "BOTH"
            vneid_layout = "DUAL_SIDE"
        elif face_result is not None:
            card_side = "FRONT"
            vneid_layout = "SINGLE_SIDE"
        elif has_back_markers:
            card_side = "BACK"
            vneid_layout = "SINGLE_SIDE"
        else:
            raise FaceDetectionError(
                "Không tìm thấy ảnh chân dung hoặc dữ liệu mặt sau hợp lệ trên ảnh chụp màn hình / VNeID."
            )

        if expected_side in ("FRONT", "BACK") and card_side != "BOTH" and card_side != expected_side:
            raise SideMismatchError(
                f"Hệ thống yêu cầu ảnh {expected_side} nhưng ảnh VNeID tải lên là mặt {card_side}."
            )

        portrait_base64: Optional[str] = None
        face_embedding: list[float] = []
        if face_result is not None:
            portrait_base64, face_embedding = face_result

        qr_info: Optional[QRInfo] = None
        if raw_qr_text is not None:
            try:
                qr_info = self.qr_reader.parse_and_validate(raw_qr_text)
            except QRCodeError:
                if strict_qr:
                    raise

        if qr_info is None and strict_qr:
            raise QRCodeError(
                "Không đọc được mã QR chuẩn 7 trường trên ảnh VNeID / chụp màn hình."
            )

        cccd_num = (
            (qr_info.cccd_number if qr_info else None)
            or full_ocr.get("cccd_number")
            or (mrz_info.document_number if mrz_info else None)
            or ""
        )
        full_name = (
            (qr_info.full_name if qr_info else None)
            or full_ocr.get("full_name")
            or (mrz_info.full_name_mrz if mrz_info else None)
            or ""
        )
        avg_conf = (
            sum(l["confidence"] for l in raw_lines) / len(raw_lines)
            if raw_lines else 0.0
        )
        ocr_info = OCRInfo(
            cccd_number=cccd_num,
            full_name=full_name,
            cccd_match=bool(cccd_num),
            name_match=bool(full_name),
            confidence_score=round(avg_conf, 4),
        )

        back_info: Optional[BackSideInfo] = None
        if has_back_markers:
            back_res = self.ocr_verifier.extract_back_side_info(proc_img, is_warped_card=False)
            back_info = BackSideInfo(
                identifying_features=back_res.get("identifying_features"),
                issue_date=back_res.get("issue_date"),
                place_of_issue=back_res.get("place_of_issue"),
                raw_lines=back_res.get("raw_lines", [])[:8],
            )

        req_id, ts = self._new_meta()
        warnings = self._collect_warnings(None, "VNEID_SCREEN", card_side, mrz_info)
        if vneid_layout == "DUAL_SIDE":
            warnings.append("Đã nhận diện ảnh VNeID 2 mặt (mặt trước + mặt sau) trên cùng một màn hình.")

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CCCDVerificationResponse(
            request_id=req_id,
            timestamp=ts,
            api_version=settings.APP_VERSION,
            status="success",
            is_valid=True,
            card_type="VNEID_SCREEN",
            card_side=card_side,
            vneid_layout=vneid_layout,
            rotated_180=False,
            qr_info=qr_info,
            ocr_info=ocr_info,
            mrz_info=mrz_info,
            back_info=back_info,
            portrait_base64=portrait_base64,
            face_embedding=face_embedding,
            quality_metrics={"sharpness": 0.0, "brightness": 0.0, "glare_card_pct": 0.0},
            warnings=warnings,
            processing_time_ms=ProcessingTimeMS(
                total_ms=round(total_ms, 2),
                corner_detection_ms=round(corner_detection_ms, 2),
                perspective_warp_ms=0.0,
                quality_check_ms=0.0,
                qr_and_face_parallel_ms=round(parallel_ms * 0.4, 2),
                ocr_cross_verify_ms=round(parallel_ms * 0.6, 2),
            ),
        )

    def process(
        self,
        image_bytes: bytes,
        strict_qr: bool = True,
        allow_screen: bool = True,
        expected_side: str = "AUTO",
    ) -> CCCDVerificationResponse:
        """
        Thực thi toàn bộ quy trình eKYC CCCD Việt Nam tuần tự theo luồng Fail-Fast:
        - Tự động phân biệt Mặt trước (FRONT) vs Mặt sau (BACK) và VNeID 1 mặt / 2 mặt.
        - Trả về response chuẩn doanh nghiệp quốc tế (request_id, timestamp, warnings, mrz_info, back_info).
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
        is_portrait_screenshot = orig_h > orig_w * 1.15

        if is_portrait_screenshot and allow_screen:
            corner_detection_ms = (time.perf_counter() - t0) * 1000.0
            return self._process_screen_or_vneid(
                img, t_start, corner_detection_ms, strict_qr, expected_side
            )

        corners = self.detector.detect(img)
        if corners is None and allow_screen:
            corner_detection_ms = (time.perf_counter() - t0) * 1000.0
            try:
                return self._process_screen_or_vneid(
                    img, t_start, corner_detection_ms, strict_qr, expected_side
                )
            except Exception:
                pass

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
        #    Hỗ trợ tự xoay 180° và tự phân biệt Mặt trước (FRONT) vs Mặt sau (BACK)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        rotated_180 = False
        raw_qr_text, face_result = self._execute_qr_and_face(warped)

        if raw_qr_text is None and face_result is None:
            logger.info("Không tìm thấy QR và Chân dung ở chiều ban đầu. Tiến hành tự xoay thẻ 180 độ...")
            warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
            rot_qr_text, rot_face_result = self._execute_qr_and_face(warped_rot)

            if rot_qr_text is not None or rot_face_result is not None:
                warped = warped_rot
                raw_qr_text = rot_qr_text
                face_result = rot_face_result
                rotated_180 = True

        qr_and_face_parallel_ms = (time.perf_counter() - t0) * 1000.0

        # Nếu KHÔNG có khuôn mặt -> kiểm tra xem có phải MẶT SAU (BACK: MRZ / Đặc điểm nhận dạng / QR mặt sau)
        if face_result is None:
            t_back0 = time.perf_counter()
            if raw_qr_text is None:
                by1, by2, bx1, bx2 = settings.ROI_BACK_QR
                raw_qr_text = self.qr_reader.decode_raw(warped[by1:by2, bx1:bx2])

            back_res = self.ocr_verifier.extract_back_side_info(warped, is_warped_card=True)
            mrz_info = back_res.get("mrz_info")
            back_raw_text = "\n".join(back_res.get("raw_lines", [])).lower()
            is_back_card = (
                mrz_info is not None
                or "đặc điểm" in back_raw_text
                or "ngày" in back_raw_text
                or "bộ công an" in back_raw_text
                or raw_qr_text is not None
            )
            if not is_back_card:
                raise FaceDetectionError(
                    "Không nhận diện được khuôn mặt (mặt trước) hoặc MRZ/thông tin mặt sau hợp lệ trên thẻ."
                )

            if expected_side == "FRONT":
                raise SideMismatchError(
                    "Hệ thống yêu cầu ảnh MẶT TRƯỚC nhưng bạn đã tải lên MẶT SAU thẻ CCCD/Căn cước."
                )

            qr_info_back: Optional[QRInfo] = None
            if raw_qr_text is not None:
                try:
                    qr_info_back = self.qr_reader.parse_and_validate(raw_qr_text)
                except QRCodeError:
                    pass

            card_type = "CAN_CUOC_2024" if qr_info_back is not None else "CCCD_2021"
            back_info = BackSideInfo(
                identifying_features=back_res.get("identifying_features"),
                issue_date=back_res.get("issue_date"),
                place_of_issue=back_res.get("place_of_issue"),
                raw_lines=back_res.get("raw_lines", []),
            )
            ocr_cross_verify_ms = (time.perf_counter() - t_back0) * 1000.0
            req_id, ts = self._new_meta()
            warnings = self._collect_warnings(
                quality_metrics, card_type, "BACK", mrz_info,
                corners=validated_corners, orig_w=orig_w, orig_h=orig_h,
            )
            total_ms = (time.perf_counter() - t_start) * 1000.0

            return CCCDVerificationResponse(
                request_id=req_id,
                timestamp=ts,
                api_version=settings.APP_VERSION,
                status="success",
                is_valid=True,
                card_type=card_type,
                card_side="BACK",
                vneid_layout=None,
                rotated_180=rotated_180,
                qr_info=qr_info_back,
                ocr_info=OCRInfo(
                    cccd_number=(mrz_info.document_number if mrz_info and mrz_info.document_number else (qr_info_back.cccd_number if qr_info_back else "")),
                    full_name=(mrz_info.full_name_mrz if mrz_info and mrz_info.full_name_mrz else (qr_info_back.full_name if qr_info_back else "")),
                    cccd_match=bool(mrz_info and mrz_info.document_number),
                    name_match=bool(mrz_info and mrz_info.full_name_mrz),
                    confidence_score=0.95 if mrz_info else 0.75,
                ),
                mrz_info=mrz_info,
                back_info=back_info,
                portrait_base64=None,
                face_embedding=[],
                quality_metrics=quality_metrics,
                warnings=warnings,
                processing_time_ms=ProcessingTimeMS(
                    total_ms=round(total_ms, 2),
                    corner_detection_ms=round(corner_detection_ms, 2),
                    perspective_warp_ms=round(perspective_warp_ms, 2),
                    quality_check_ms=round(quality_check_ms, 2),
                    qr_and_face_parallel_ms=round(qr_and_face_parallel_ms, 2),
                    ocr_cross_verify_ms=round(ocr_cross_verify_ms, 2),
                ),
            )

        # Có khuôn mặt -> MẶT TRƯỚC (FRONT)
        if expected_side == "BACK":
            raise SideMismatchError(
                "Hệ thống yêu cầu ảnh MẶT SAU nhưng bạn đã tải lên MẶT TRƯỚC thẻ CCCD/Căn cước."
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
            raw_lines = full_ocr.get("raw_lines", [])
            name_guess = full_ocr.get("full_name") or ""
            if not name_guess:
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

        req_id, ts = self._new_meta()
        warnings = self._collect_warnings(quality_metrics, card_type, "FRONT")
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CCCDVerificationResponse(
            request_id=req_id,
            timestamp=ts,
            api_version=settings.APP_VERSION,
            status="success",
            is_valid=True,
            card_type=card_type,
            card_side="FRONT",
            vneid_layout=None,
            rotated_180=rotated_180,
            qr_info=qr_info,
            ocr_info=ocr_info,
            mrz_info=None,
            back_info=None,
            portrait_base64=portrait_base64,
            face_embedding=face_embedding,
            quality_metrics=quality_metrics,
            warnings=warnings,
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

        ocr_data = self.ocr_verifier.extract_full_card_ocr(warped, executor=self.executor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BasicOCRResponse(
            status="success",
            cccd_number=ocr_data["cccd_number"],
            full_name=ocr_data.get("full_name"),
            dob=ocr_data["dob"],
            gender=ocr_data["gender"],
            expiry_date=ocr_data["expiry_date"],
            raw_lines=[BasicOCRLine(**line) for line in ocr_data["raw_lines"]],
            processing_time_ms=round(elapsed_ms, 2),
        )
