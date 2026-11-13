import logging
import re
import unicodedata
from typing import Optional, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from app.core.config import settings
from app.core.exceptions import CrossVerificationError
from app.schemas.cccd import OCRInfo, QRInfo

logger = logging.getLogger(__name__)


def strip_accents(text: str) -> str:
    """
    Chuẩn hóa Unicode NFC và loại bỏ toàn bộ dấu tiếng Việt.
    Đặc biệt chuyển đổi chữ Đ/đ -> D/d.
    """
    if not text:
        return ""
    # Chuyển Đ/đ thành D/d trước khi phân rã tổ hợp
    text = text.replace("Đ", "D").replace("đ", "d")
    # Phân rã ký tự có dấu thành ký tự gốc + dấu (NFD)
    nfd_text = unicodedata.normalize("NFD", text)
    # Loại bỏ các ký tự dấu (Nonspacing Mark 'Mn')
    stripped = "".join(c for c in nfd_text if unicodedata.category(c) != "Mn")
    # Trả về chuỗi NFC
    return unicodedata.normalize("NFC", stripped)


def normalize_name(text: str) -> str:
    """
    Chuẩn hóa họ tên: bỏ dấu tiếng Việt, viết hoa, loại bỏ ký tự đặc biệt thừa,
    và chuẩn hóa khoảng trắng đơn.
    """
    s = strip_accents(text).upper()
    # Chỉ giữ chữ cái A-Z và khoảng trắng
    s = re.sub(r"[^A-Z\s]", " ", s)
    return " ".join(s.split())


class OCRVerifier:
    """
    Module OCR chuyên biệt với RapidOCR (chỉ chạy Text Recognition - use_rec=True,
    tắt detection và classification) trên 2 vùng ROI Số CCCD và Họ tên.
    Thực hiện đối chiếu chéo (Cross-Verification) chống làm giả / chỉnh sửa thẻ.
    """

    def __init__(self):
        self.engine: Optional[RapidOCR] = None
        self._init_ocr()

    def _init_ocr(self) -> None:
        try:
            self.engine = RapidOCR(
                det_use_cuda=False,
                cls_use_cuda=False,
                rec_use_cuda=False
            )
            logger.info("Khởi tạo RapidOCR thành công.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo RapidOCR: {e}")
            raise

    def recognize_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện văn bản trên ảnh cắt ROI với use_det=False, use_cls=False, use_rec=True.
        Trả về (text, confidence).
        """
        if self.engine is None:
            raise RuntimeError("RapidOCR engine chưa sẵn sàng.")

        # Tiền xử lý nhẹ: Grayscale & Tăng tương phản nếu cần
        # RapidOCR nhận BGR hoặc RGB
        results, _ = self.engine(crop_bgr, use_det=False, use_cls=False, use_rec=True)

        if not results:
            return "", 0.0

        texts = []
        confidences = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                t = str(item[0]).strip()
                score = float(item[1])
                if t:
                    texts.append(t)
                    confidences.append(score)

        merged_text = " ".join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return merged_text, avg_conf

    def verify_and_cross_check(
        self,
        warped_card: np.ndarray,
        qr_info: QRInfo,
        executor=None
    ) -> OCRInfo:
        """
        Cắt 2 vùng ROI Số CCCD và Họ tên trên thẻ 1000 x 630:
        1. Vùng Số CCCD: y: 245..315, x: 390..760
        2. Vùng Họ tên:   y: 320..395, x: 385..920
        Chạy RapidOCR và đối chiếu chéo với dữ liệu mã QR:
        - Số CCCD phải trùng 100% với 12 số trong QR.
        - Họ tên không dấu (NFC) phải trùng khớp với họ tên trong QR.
        """
        # 1. Cắt ROI
        ny1, ny2, nx1, nx2 = settings.ROI_CCCD_NUMBER
        crop_number = warped_card[ny1:ny2, nx1:nx2]

        my1, my2, mx1, mx2 = settings.ROI_FULL_NAME
        crop_name = warped_card[my1:my2, mx1:mx2]

        # 2. OCR song song nếu có executor
        if executor is not None:
            future_num = executor.submit(self.recognize_crop, crop_number)
            future_name = executor.submit(self.recognize_crop, crop_name)
            raw_num_text, conf_num = future_num.result()
            raw_name_text, conf_name = future_name.result()
        else:
            raw_num_text, conf_num = self.recognize_crop(crop_number)
            raw_name_text, conf_name = self.recognize_crop(crop_name)

        # Trích xuất toàn bộ chữ số
        ocr_digits = "".join(re.findall(r"\d", raw_num_text))
        avg_conf = (conf_num + conf_name) / 2.0 if (conf_num > 0 or conf_name > 0) else 0.0

        # 4. Đối chiếu chéo Số CCCD (phải khớp 100% 12 số)
        qr_cccd = qr_info.cccd_number.strip()
        cccd_match = (ocr_digits == qr_cccd)

        if not cccd_match:
            raise CrossVerificationError(
                f"Số CCCD nhận diện qua OCR ('{ocr_digits}') không trùng khớp 100% với dữ liệu mã QR ('{qr_cccd}'). "
                "Cảnh báo: Nghi ngờ phôi thẻ đã bị tẩy xóa, dán đè số hoặc làm giả!"
            )

        # 5. Đối chiếu chéo Họ và Tên (chuẩn hóa NFC, bỏ dấu tiếng Việt)
        norm_ocr_name = normalize_name(raw_name_text)
        norm_qr_name = normalize_name(qr_info.full_name)

        # So khớp họ tên: chấp nhận nếu trùng khớp hoàn toàn chuỗi chuẩn hóa
        # hoặc là một phần họ tên khớp nhau (ví dụ tên dài bị che khuất nhẹ)
        name_match = (norm_ocr_name == norm_qr_name)
        if not name_match:
            # Thử kiểm tra khoảng cách ký tự hoặc trường hợp thiếu từ đệm
            ocr_words = set(norm_ocr_name.split())
            qr_words = set(norm_qr_name.split())
            common_words = ocr_words.intersection(qr_words)
            # Nếu ít nhất 70% số từ trùng nhau hoặc chứa họ + tên chính
            if len(qr_words) > 0 and len(common_words) / len(qr_words) >= 0.70:
                name_match = True
            else:
                raise CrossVerificationError(
                    f"Họ tên nhận diện qua OCR ('{norm_ocr_name}') không trùng khớp với mã QR ('{norm_qr_name}'). "
                    "Cảnh báo: Nghi ngờ họ tên trên mặt thẻ bị chỉnh sửa khác biệt so với mã QR!"
                )

        return OCRInfo(
            cccd_number=ocr_digits,
            full_name=raw_name_text,
            cccd_match=cccd_match,
            name_match=name_match,
            confidence_score=round(avg_conf, 4),
        )
