import abc
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.core.exceptions import GeometricValidationError

logger = logging.getLogger(__name__)


def sort_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 đỉnh của tứ giác theo thứ tự chuẩn:
    [top_left, top_right, bottom_right, bottom_left]
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    # Sắp xếp theo trục Y để chia 2 điểm trên và 2 điểm dưới
    y_idx = np.argsort(pts[:, 1])
    top_two = pts[y_idx[:2]]
    bottom_two = pts[y_idx[2:]]

    # Điểm có X nhỏ hơn ở nửa trên là Top-Left, lớn hơn là Top-Right
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    # Điểm có X lớn hơn ở nửa dưới là Bottom-Right, nhỏ hơn là Bottom-Left
    bl, br = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def validate_geometry(corners: Optional[np.ndarray], orig_w: int, orig_h: int) -> np.ndarray:
    """
    Kiểm tra các tiêu chuẩn hình học khắt khe của thẻ CCCD.
    Fail-fast: Ném ngoại lệ GeometricValidationError nếu vi phạm.
    """
    if corners is None or len(corners) != 4:
        raise GeometricValidationError(
            "Không thể phát hiện đủ 4 góc của thẻ CCCD. Vui lòng đặt thẻ trên mặt phẳng tương phản và chụp rõ nét."
        )

    corners = sort_corners(corners)
    tl, tr, br, bl = corners

    # 1. Kiểm tra góc sát mép ảnh dưới 8px (lỗi cắt góc / chạm viền)
    margin = settings.MIN_CORNER_BORDER_DISTANCE
    for i, (x, y) in enumerate(corners):
        if x < margin or y < margin or x > (orig_w - margin) or y > (orig_h - margin):
            raise GeometricValidationError(
                f"Góc thẻ thứ {i+1} tọa độ ({x:.0f}, {y:.0f}) nằm quá sát mép ảnh (dưới {margin}px). "
                "Có nguy cơ mất góc hoặc chạm viền thẻ, vui lòng căn chỉnh thẻ vào giữa khung hình."
            )

    # 2. Kiểm tra tỷ lệ diện tích thẻ / toàn ảnh
    card_area = float(cv2.contourArea(corners))
    total_area = float(orig_w * orig_h)
    area_ratio = card_area / total_area if total_area > 0 else 0.0

    if area_ratio < settings.MIN_CARD_AREA_RATIO:
        raise GeometricValidationError(
            f"Thẻ CCCD ở quá xa camera (diện tích thẻ chiếm {area_ratio*100:.1f}% < "
            f"{settings.MIN_CARD_AREA_RATIO*100:.0f}% ảnh). Vui lòng đưa camera lại gần thẻ hơn."
        )
    if area_ratio > settings.MAX_CARD_AREA_RATIO:
        raise GeometricValidationError(
            f"Thẻ CCCD ở quá gần camera (diện tích thẻ chiếm {area_ratio*100:.1f}% > "
            f"{settings.MAX_CARD_AREA_RATIO*100:.0f}% ảnh). Vui lòng lùi camera lại một chút để lấy trọn viền thẻ."
        )

    # 3. Kiểm tra tỷ lệ chiều rộng / chiều cao (aspect ratio chuẩn ID-1 = 1.586)
    width_top = float(np.linalg.norm(tr - tl))
    width_bot = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))

    avg_width = (width_top + width_bot) / 2.0
    avg_height = (height_left + height_right) / 2.0

    if avg_height <= 0:
        raise GeometricValidationError("Chiều cao thẻ không hợp lệ (bằng 0).")

    aspect_ratio = avg_width / avg_height
    if aspect_ratio < settings.MIN_CARD_ASPECT_RATIO or aspect_ratio > settings.MAX_CARD_ASPECT_RATIO:
        raise GeometricValidationError(
            f"Tỷ lệ chiều rộng/chiều cao thẻ ({aspect_ratio:.2f}) nằm ngoài giới hạn cho phép "
            f"[{settings.MIN_CARD_ASPECT_RATIO:.2f}, {settings.MAX_CARD_ASPECT_RATIO:.2f}] so với chuẩn ID-1 (1.586). "
            "Thẻ bị chụp quá nghiêng, méo hoặc sai định dạng."
        )

    return corners


def warp_card(
    image: np.ndarray,
    corners: np.ndarray,
    target_w: int = settings.CARD_WARP_WIDTH,
    target_h: int = settings.CARD_WARP_HEIGHT
) -> np.ndarray:
    """
    Nắn phẳng phối cảnh (Perspective Warp) đưa thẻ về kích thước chuẩn cố định 1000 x 630.
    """
    src_pts = corners.astype(np.float32)
    dst_pts = np.array([
        [0.0, 0.0],
        [float(target_w), 0.0],
        [float(target_w), float(target_h)],
        [0.0, float(target_h)]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped


class BaseCardDetector(abc.ABC):
    """Giao diện trừu tượng chuẩn cho các bộ bắt góc thẻ (ONNX / OpenCV / YOLO / DocAligner)."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Nhận ảnh BGR gốc, trả về 4 tọa độ góc theo hệ quy chiếu ảnh gốc.
        Shape: (4, 2) hoặc None nếu không phát hiện được.
        """
        pass


class ContourCardDetector(BaseCardDetector):
    """
    Bộ phát hiện góc thẻ bằng Contour & Polygon Approximation tối ưu cực nhanh (< 3ms trên CPU).
    Tự động tiền xử lý đa ngưỡng (Canny + Otsu Adaptive) để tìm hình tứ giác lồi chuẩn thẻ ID-1.
    """

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        orig_h, orig_w = image.shape[:2]
        max_dim = settings.CORNER_DETECT_RESIZE_MAX

        # Resize ảnh về cạnh dài max_dim để tăng tốc độ
        scale = max_dim / float(max(orig_h, orig_w))
        if scale < 1.0:
            resized = cv2.resize(
                image,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            resized = image
            scale = 1.0

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thử 2 chiến lược: Canny Edges và Adaptive Threshold
        strategies = [
            lambda: cv2.Canny(blurred, 30, 150),
            lambda: cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            ),
        ]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        resized_area = resized.shape[0] * resized.shape[1]

        best_poly = None
        best_score = -1.0

        for get_binary in strategies:
            binary = get_binary()
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                area = cv2.contourArea(cnt)
                if area < resized_area * 0.04:
                    continue

                peri = cv2.arcLength(cnt, True)
                for eps in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2)
                        sorted_pts = sort_corners(pts)
                        # Tính aspect ratio sơ bộ
                        w = np.linalg.norm(sorted_pts[1] - sorted_pts[0])
                        h = np.linalg.norm(sorted_pts[3] - sorted_pts[0])
                        if h > 0:
                            ar = w / h
                            if 1.25 <= ar <= 1.95:
                                # Chuẩn ID-1: 1.586, tính điểm lệch
                                diff = abs(ar - 1.586)
                                score = area / (1.0 + diff * 1000.0)
                                if score > best_score:
                                    best_score = score
                                    best_poly = sorted_pts
                        break

            if best_poly is not None:
                break

        if best_poly is not None:
            # Quy đổi tọa độ về kích thước ảnh gốc
            orig_corners = best_poly / scale
            return orig_corners

        return None


class ONNXDocAlignerDetector(BaseCardDetector):
    """
    Interface chuẩn chạy model ONNX DocAligner (LCNet100 / FastViT / YOLO11n-pose).
    Chạy trực tiếp qua ONNXRuntime CPUExecutionProvider.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self._init_session()

    def _init_session(self) -> None:
        if not self.model_path.exists():
            logger.warning(f"ONNX DocAligner model không tìm thấy tại {self.model_path}")
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Đã nạp ONNX DocAligner model từ {self.model_path}")

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.session is None:
            return None

        orig_h, orig_w = image.shape[:2]
        # Tiền xử lý về [1, 3, 256, 256] RGB chuẩn hóa [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # Chạy inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        heatmaps = outputs[0][0]  # Shape (4, 128, 128)

        corners = []
        for i in range(4):
            hm = heatmaps[i]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            # Tọa độ quy đổi về ảnh gốc
            px = float(x) * (orig_w / 128.0)
            py = float(y) * (orig_h / 128.0)
            corners.append([px, py])

        corners_np = np.array(corners, dtype=np.float32)
        return sort_corners(corners_np)


class HybridCardDetector(BaseCardDetector):
    """
    Bộ bắt góc thẻ lai:
    1. Ưu tiên ContourCardDetector vì siêu nhanh (~2ms).
    2. Nếu không tìm thấy, fallback sang ONNXDocAlignerDetector (~35ms).
    """

    def __init__(self, onnx_model_path: Optional[Path] = None):
        self.fast_contour = ContourCardDetector()
        self.onnx_detector: Optional[ONNXDocAlignerDetector] = None
        if onnx_model_path and onnx_model_path.exists():
            self.onnx_detector = ONNXDocAlignerDetector(onnx_model_path)

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Thử contour trước để tối ưu latency < 60ms
        corners = self.fast_contour.detect(image)
        if corners is not None:
            return corners

        # Fallback sang ONNX nếu contour không ra
        if self.onnx_detector is not None:
            corners = self.onnx_detector.detect(image)
            if corners is not None:
                return corners

        return None
