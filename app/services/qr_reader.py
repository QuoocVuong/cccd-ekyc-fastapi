import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import zxingcpp

from app.core.config import settings
from app.core.exceptions import QRCodeError
from app.schemas.cccd import QRInfo

logger = logging.getLogger(__name__)


class QRReader:
    """
    Module giải mã và xác thực mã QR trên thẻ CCCD Việt Nam.
    Sử dụng WeChatQRCode làm bộ giải mã chính (nhanh & chịu góc méo tốt),
    và zxing-cpp làm bộ giải mã dự phòng.
    """

    def __init__(self, models_dir: Path = settings.WECHAT_QR_DIR):
        self.models_dir = models_dir
        self.wechat_detector: Optional[cv2.wechat_qrcode.WeChatQRCode] = None
        self._init_wechat_qr()

    def _init_wechat_qr(self) -> None:
        """Khởi tạo WeChatQRCode với 4 file model CNN."""
        detect_prototxt = self.models_dir / "detect.prototxt"
        detect_caffe = self.models_dir / "detect.caffemodel"
        sr_prototxt = self.models_dir / "sr.prototxt"
        sr_caffe = self.models_dir / "sr.caffemodel"

        if (
            detect_prototxt.exists()
            and detect_caffe.exists()
            and sr_prototxt.exists()
            and sr_caffe.exists()
            and hasattr(cv2, "wechat_qrcode_WeChatQRCode")
        ):
            try:
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
                    str(detect_prototxt),
                    str(detect_caffe),
                    str(sr_prototxt),
                    str(sr_caffe),
                )
                logger.info("Khởi tạo WeChatQRCode thành công với 4 model files.")
            except Exception as e:
                logger.warning(f"Lỗi khởi tạo WeChatQRCode: {e}. Sẽ dùng fallback zxing-cpp.")
        else:
            logger.warning(
                "Không tìm thấy đủ 4 file model WeChatQRCode. Sẽ ưu tiên zxing-cpp."
            )

    def decode_raw(self, qr_roi: np.ndarray) -> Optional[str]:
        """
        Thực hiện giải mã trên ảnh cắt ROI vùng mã QR.
        Thử WeChatQR trước, nếu không ra thì fallback sang zxingcpp.
        """
        # 1. Thử WeChatQRCode (chính)
        if self.wechat_detector is not None:
            try:
                results, _ = self.wechat_detector.detectAndDecode(qr_roi)
                for res in results:
                    if res and len(res.strip()) > 0:
                        return res.strip()
            except Exception as e:
                logger.debug(f"WeChatQR decode error: {e}")

        # 2. Thử zxingcpp (dự phòng)
        try:
            barcodes = zxingcpp.read_barcodes(qr_roi)
            for b in barcodes:
                if b.text and len(b.text.strip()) > 0:
                    return b.text.strip()
        except Exception as e:
            logger.debug(f"zxingcpp decode error: {e}")

        # 3. Thử tiền xử lý nhẹ (tăng tương phản / grayscale) nếu cả 2 chưa đọc được
        try:
            gray = cv2.cvtColor(qr_roi, cv2.COLOR_BGR2GRAY)
            # Thử lại zxingcpp trên ảnh xám
            barcodes = zxingcpp.read_barcodes(gray)
            for b in barcodes:
                if b.text and len(b.text.strip()) > 0:
                    return b.text.strip()
        except Exception:
            pass

        return None

    def parse_and_validate(self, raw_text: str) -> QRInfo:
        """
        Kiểm tra và bóc tách dữ liệu chuỗi QR CCCD theo đúng chuẩn 7 trường:
        cccd_number (12 số) | old_cmnd | full_name | dob (DDMMYYYY) | gender | address | issue_date (DDMMYYYY)
        """
        parts = raw_text.split("|")
        if len(parts) != 7:
            raise QRCodeError(
                f"Chuỗi mã QR không đúng định dạng 7 trường của CCCD Việt Nam (tìm thấy {len(parts)} trường thay vì 7)."
            )

        cccd_number = parts[0].strip()
        old_cmnd = parts[1].strip() or None
        full_name = parts[2].strip()
        dob = parts[3].strip()
        gender = parts[4].strip()
        address = parts[5].strip()
        issue_date = parts[6].strip()

        # 1. Validate Số CCCD (phải đúng 12 chữ số)
        if not re.fullmatch(r"\d{12}", cccd_number):
            raise QRCodeError(
                f"Số CCCD trong mã QR '{cccd_number}' không đúng định dạng 12 chữ số."
            )

        # 2. Validate Ngày sinh (DDMMYYYY - 8 chữ số)
        if not re.fullmatch(r"\d{8}", dob):
            raise QRCodeError(
                f"Ngày sinh trong mã QR '{dob}' không đúng định dạng DDMMYYYY."
            )

        # 3. Validate Ngày cấp (DDMMYYYY - 8 chữ số)
        if not re.fullmatch(r"\d{8}", issue_date):
            raise QRCodeError(
                f"Ngày cấp trong mã QR '{issue_date}' không đúng định dạng DDMMYYYY."
            )

        # 4. Validate Họ tên không rỗng
        if not full_name:
            raise QRCodeError("Họ tên trong mã QR bị rỗng.")

        return QRInfo(
            cccd_number=cccd_number,
            old_cmnd=old_cmnd,
            full_name=full_name,
            dob=dob,
            gender=gender,
            address=address,
            issue_date=issue_date,
        )
