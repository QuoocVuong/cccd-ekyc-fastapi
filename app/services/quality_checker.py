import logging
from typing import Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import ImageQualityError

logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Module kiểm tra chất lượng ảnh thẻ và chống giả mạo (Fail-Fast Anti-Spoofing):
    1. Chống mờ / nhòe nét (Laplacian Variance).
    2. Chống lóa đèn flash trên toàn thẻ, vùng QR, và vùng ảnh chân dung.
    3. Chống chụp lại qua màn hình (Screen Replay / Moiré pattern qua 2D FFT).
    """

    @staticmethod
    def check_sharpness(gray_card: np.ndarray) -> float:
        """
        Tính toán độ sắc nét bằng phương sai toán tử Laplacian.
        Nếu var < 85.0 -> Ném ImageQualityError.
        """
        laplacian = cv2.Laplacian(gray_card, cv2.CV_64F)
        score = float(laplacian.var())

        if score < settings.MIN_SHARPNESS_LAPLACIAN_VAR:
            raise ImageQualityError(
                f"Ảnh thẻ CCCD bị mờ, nhòe nét (điểm độ nét: {score:.1f} < "
                f"{settings.MIN_SHARPNESS_LAPLACIAN_VAR:.1f}). Vui lòng giữ chắc tay và chụp lại rõ ràng."
            )
        return score

    @staticmethod
    def check_glare(card_bgr: np.ndarray) -> dict[str, float]:
        """
        Kiểm tra lóa đèn flash bằng hệ màu HSV (V > 245 và S < 30).
        Kiểm tra 3 vùng: Toàn thẻ, vùng QR, và vùng Chân dung.
        """
        hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Mask pixel bị lóa sáng trắng
        glare_mask = (val > settings.GLARE_V_THRESHOLD) & (sat < settings.GLARE_S_THRESHOLD)

        # 1. Toàn thẻ (1000 x 630)
        total_pixels = float(card_bgr.shape[0] * card_bgr.shape[1])
        card_glare_ratio = float(np.count_nonzero(glare_mask)) / total_pixels

        if card_glare_ratio > settings.MAX_GLARE_CARD_RATIO:
            raise ImageQualityError(
                f"Ảnh bị lóa sáng đèn flash trên toàn thẻ ({card_glare_ratio * 100:.2f}% > "
                f"{settings.MAX_GLARE_CARD_RATIO * 100:.1f}% diện tích). Vui lòng tắt flash hoặc đổi góc chụp."
            )

        # 2. Vùng mã QR (y: 10..290, x: 680..990)
        qy1, qy2, qx1, qx2 = settings.ROI_QR
        qr_mask = glare_mask[qy1:qy2, qx1:qx2]
        qr_pixels = float(qr_mask.shape[0] * qr_mask.shape[1])
        qr_glare_ratio = float(np.count_nonzero(qr_mask)) / qr_pixels if qr_pixels > 0 else 0.0

        if qr_glare_ratio > settings.MAX_GLARE_QR_RATIO:
            raise ImageQualityError(
                f"Vùng mã QR bị lóa đèn flash ({qr_glare_ratio * 100:.2f}% > "
                f"{settings.MAX_GLARE_QR_RATIO * 100:.1f}%). Vui lòng đổi góc chụp để tránh mất nét mã QR."
            )

        # 3. Vùng ảnh chân dung (y: 140..580, x: 30..370)
        fy1, fy2, fx1, fx2 = settings.ROI_PORTRAIT
        face_mask = glare_mask[fy1:fy2, fx1:fx2]
        face_pixels = float(face_mask.shape[0] * face_mask.shape[1])
        face_glare_ratio = float(np.count_nonzero(face_mask)) / face_pixels if face_pixels > 0 else 0.0

        if face_glare_ratio > settings.MAX_GLARE_PORTRAIT_RATIO:
            raise ImageQualityError(
                f"Vùng ảnh chân dung bị lóa đèn flash ({face_glare_ratio * 100:.2f}% > "
                f"{settings.MAX_GLARE_PORTRAIT_RATIO * 100:.1f}%). Vui lòng căn chỉnh ánh sáng tránh lóa khuôn mặt."
            )

        # 4. Vùng thông tin chữ (Số CCCD, Họ tên, Ngày sinh...)
        ty1, ty2, tx1, tx2 = settings.ROI_TEXT_ZONE
        text_mask = glare_mask[ty1:ty2, tx1:tx2]
        text_pixels = float(text_mask.shape[0] * text_mask.shape[1])
        text_glare_ratio = float(np.count_nonzero(text_mask)) / text_pixels if text_pixels > 0 else 0.0

        if text_glare_ratio > settings.MAX_GLARE_TEXT_RATIO:
            raise ImageQualityError(
                f"Vùng thông tin chữ trên thẻ bị lóa sáng ({text_glare_ratio * 100:.2f}% > "
                f"{settings.MAX_GLARE_TEXT_RATIO * 100:.1f}%). Vui lòng đổi góc chụp để tránh chói chữ."
            )

        return {
            "card_glare_ratio": card_glare_ratio,
            "qr_glare_ratio": qr_glare_ratio,
            "portrait_glare_ratio": face_glare_ratio,
            "text_glare_ratio": text_glare_ratio,
        }

    @staticmethod
    def check_card_occlusion(card_bgr: np.ndarray, gray_card: np.ndarray) -> None:
        """
        Kiểm tra thẻ bị lẹm góc, mất góc (vùng đen do warp ngoài biên) hoặc bị che thẻ (ngón tay/vật cản).
        """
        h, w = gray_card.shape
        patch = 60
        corners_patches = [
            ("trên-trái", gray_card[0:patch, 0:patch]),
            ("trên-phải", gray_card[0:patch, w - patch : w]),
            ("dưới-phải", gray_card[h - patch : h, w - patch : w]),
            ("dưới-trái", gray_card[h - patch : h, 0:patch]),
        ]
        for corner_name, region in corners_patches:
            black_ratio = float(np.count_nonzero(region < 12)) / float(region.size)
            if black_ratio > 0.30 or float(np.mean(region)) < 25.0:
                raise ImageQualityError(
                    f"Phát hiện góc {corner_name} của thẻ bị lẹm, cắt góc hoặc bị che khuất. "
                    "Vui lòng chụp trọn vẹn cả 4 góc thẻ."
                )

        # Kiểm tra vùng thông tin chữ có bị che bởi ngón tay / vật tối màu / đồng nhất bất thường
        ty1, ty2, tx1, tx2 = settings.ROI_TEXT_ZONE
        text_gray = gray_card[ty1:ty2, tx1:tx2]
        if float(np.mean(text_gray)) < 28.0 or float(np.std(text_gray)) < 4.5:
            raise ImageQualityError(
                "Phát hiện vùng thông tin trên thẻ bị che khuất hoặc quá đồng nhất không đọc được chữ. "
                "Vui lòng không che thẻ bằng tay hoặc vật cản."
            )

    @staticmethod
    def check_screen_replay(gray_card: np.ndarray) -> float:
        """
        Kiểm tra bất thường phổ tần số cao bằng 2D FFT (Moiré pattern / Screen Replay).
        Các bức ảnh chụp lại từ màn hình LCD/OLED chứa lưới pixel tuần hoàn sinh ra
        đỉnh nhọn bất thường trong phổ tần số cao.
        """
        if not settings.ENABLE_MOIRE_CHECK:
            return 0.0

        # Lấy vùng trung tâm thẻ để tránh hiệu ứng cạnh biên của thẻ tạo đường chữ thập trong FFT
        h, w = gray_card.shape
        crop = gray_card[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]

        # Biến đổi Fourier 2D
        f = np.fft.fft2(crop.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        ch, cw = magnitude.shape
        cy, cx = ch // 2, cw // 2
        y, x = np.ogrid[:ch, :cw]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Loại bỏ vùng tần số thấp ở tâm (bán kính r = 30)
        hf_mask = dist_from_center > 30.0
        if not np.any(hf_mask):
            return 0.0

        hf_values = magnitude[hf_mask]
        mean_val = float(np.mean(hf_values))
        std_val = float(np.std(hf_values))
        max_val = float(np.max(hf_values))

        # Peak-to-Mean Z-score
        peak_ratio = (max_val - mean_val) / (std_val + 1e-6)

        # Ngưỡng phát hiện moiré màn hình
        threshold = settings.MOIRE_MAX_PEAK_RATIO
        if peak_ratio > threshold:
            raise ImageQualityError(
                f"Nghi ngờ thẻ CCCD được chụp lại qua màn hình thiết bị điện tử "
                f"(phát hiện vân Moiré tần số cao, chỉ số: {peak_ratio:.1f} > {threshold:.1f}). "
                "Vui lòng chụp trực tiếp thẻ cứng gốc."
            )

        return peak_ratio

    @staticmethod
    def check_brightness(gray_card: np.ndarray) -> float:
        """Kiểm tra độ sáng trung bình của thẻ (tránh quá tối hoặc cháy sáng toàn cục)."""
        mean_val = float(np.mean(gray_card))
        if mean_val < settings.MIN_BRIGHTNESS_MEAN:
            raise ImageQualityError(
                f"Ảnh thẻ quá tối (độ sáng: {mean_val:.1f} < {settings.MIN_BRIGHTNESS_MEAN:.1f}). "
                "Vui lòng chụp ở nơi có đủ ánh sáng."
            )
        if mean_val > settings.MAX_BRIGHTNESS_MEAN:
            raise ImageQualityError(
                f"Ảnh thẻ bị cháy sáng quá mức (độ sáng: {mean_val:.1f} > {settings.MAX_BRIGHTNESS_MEAN:.1f}). "
                "Vui lòng giảm nguồn sáng hoặc tránh ánh đèn chiếu trực tiếp."
            )
        return mean_val

    @classmethod
    def validate_card_quality(cls, card_bgr: np.ndarray) -> dict[str, float]:
        """
        Chạy toàn bộ chuỗi kiểm tra chất lượng. Fail-fast khi phát hiện bất kỳ lỗi nào.
        Trả về các chỉ số đo đạc để phục vụ debug/tinh chỉnh.
        """
        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
        cls.check_card_occlusion(card_bgr, gray)
        sharpness = cls.check_sharpness(gray)
        brightness = cls.check_brightness(gray)
        glare_metrics = cls.check_glare(card_bgr)
        moire_score = cls.check_screen_replay(gray)
        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "glare_card_pct": round(glare_metrics["card_glare_ratio"] * 100.0, 2),
            "glare_qr_pct": round(glare_metrics["qr_glare_ratio"] * 100.0, 2),
            "glare_portrait_pct": round(glare_metrics["portrait_glare_ratio"] * 100.0, 2),
            "glare_text_pct": round(glare_metrics["text_glare_ratio"] * 100.0, 2),
            "moire_score": round(moire_score, 2),
        }
