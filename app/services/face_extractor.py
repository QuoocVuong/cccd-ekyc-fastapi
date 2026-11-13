import base64
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings
from app.core.exceptions import FaceDetectionError

logger = logging.getLogger(__name__)


class FaceExtractor:
    """
    Module phát hiện khuôn mặt, cắt ảnh chân dung chuẩn tỷ lệ 3x4 và trích xuất vector embedding 512 chiều.
    Sử dụng InsightFace (buffalo_sc: det_500m + w600k_mbf) chạy tối ưu trên CPUExecutionProvider.
    """

    def __init__(self, model_name: str = settings.INSIGHTFACE_NAME, root_dir: Path = settings.BASE_DIR):
        self.model_name = model_name
        self.root_dir = root_dir
        self.app: Optional[FaceAnalysis] = None
        self._init_face_analysis()

    def _init_face_analysis(self) -> None:
        """Khởi tạo FaceAnalysis với cấu hình tối ưu CPU."""
        try:
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(self.root_dir),
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            # Chuẩn bị model với det_size cố định (320, 320) trên ROI nhỏ giúp đạt latency < 25ms
            self.app.prepare(ctx_id=-1, det_size=settings.INSIGHTFACE_DET_SIZE)
            logger.info("Khởi tạo InsightFace (buffalo_sc) thành công trên CPU.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo InsightFace: {e}")
            raise

    def process(
        self, card_image: np.ndarray, face_roi: np.ndarray, roi_offset: Tuple[int, int]
    ) -> Optional[Tuple[str, list[float]]]:
        """
        Nhận diện khuôn mặt trên vùng face_roi:
        - Chọn khuôn mặt lớn nhất có det_score >= 0.65.
        - Mở rộng bbox (crop_w = fw * 1.60, crop_h = fh * 2.10, top_margin = 0.45 * crop_h).
        - Cắt ảnh chân dung từ card_image gốc và encode Base64 JPEG.
        - Trích xuất normed_embedding (512 chiều).
        Trả về (portrait_base64, face_embedding) hoặc None nếu không tìm thấy.
        """
        if self.app is None:
            raise FaceDetectionError("InsightFace model chưa được khởi tạo.")

        faces = self.app.get(face_roi)
        if not faces:
            return None

        # Lọc các khuôn mặt đạt ngưỡng tin cậy >= 0.65
        valid_faces = [
            f for f in faces if getattr(f, "det_score", 0.0) >= settings.FACE_DET_SCORE_THRESHOLD
        ]
        if not valid_faces:
            return None

        # Lấy khuôn mặt có diện tích lớn nhất trong vùng chân dung
        def get_face_area(face) -> float:
            b = face.bbox
            return float((b[2] - b[0]) * (b[3] - b[1]))

        best_face = max(valid_faces, key=get_face_area)

        # Tọa độ bounding box trong face_roi
        fx1, fy1, fx2, fy2 = best_face.bbox.astype(int)
        fw = fx2 - fx1
        fh = fy2 - fy1

        if fw <= 0 or fh <= 0:
            return None

        # Quy đổi tọa độ về ảnh thẻ gốc (1000 x 630)
        roi_y1, roi_x1 = roi_offset
        card_fx1 = fx1 + roi_x1
        card_fy1 = fy1 + roi_y1
        cx = card_fx1 + fw // 2

        # Tính kích thước crop mở rộng chuẩn ảnh thẻ 3x4
        crop_w = int(fw * settings.FACE_CROP_WIDTH_SCALE)
        crop_h = int(fh * settings.FACE_CROP_HEIGHT_SCALE)

        # Ưu tiên khoảng trống phía trên đỉnh đầu 0.45 * crop_h
        crop_y1 = card_fy1 - int(crop_h * settings.FACE_TOP_HEAD_MARGIN_RATIO)
        crop_y2 = crop_y1 + crop_h
        crop_x1 = cx - crop_w // 2
        crop_x2 = crop_x1 + crop_w

        # Giới hạn trong kích thước thẻ
        card_h, card_w = card_image.shape[:2]
        crop_y1 = max(0, crop_y1)
        crop_y2 = min(card_h, crop_y2)
        crop_x1 = max(0, crop_x1)
        crop_x2 = min(card_w, crop_x2)

        portrait_img = card_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if portrait_img.size == 0:
            return None

        # Encode JPEG Base64
        success, buffer = cv2.imencode(".jpg", portrait_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise FaceDetectionError("Không thể mã hóa ảnh chân dung sang định dạng JPEG.")

        b64_str = base64.b64encode(buffer).decode("utf-8")
        portrait_base64 = f"data:image/jpeg;base64,{b64_str}"

        # Lấy embedding 512 chiều chuẩn hóa L2
        normed_embedding = best_face.normed_embedding
        if normed_embedding is None or len(normed_embedding) == 0:
            raise FaceDetectionError("Không thể trích xuất vector embedding khuôn mặt.")

        embedding_list = normed_embedding.astype(float).tolist()
        return portrait_base64, embedding_list

    def extract_selfie_embedding(self, selfie_bgr: np.ndarray) -> Tuple[list[float], float]:
        """
        Trích xuất vector embedding 512 chiều từ ảnh Selfie / ảnh chân dung đối chiếu.
        Trả về (embedding_list, det_score).
        """
        if self.app is None:
            raise FaceDetectionError("InsightFace model chưa được khởi tạo.")

        faces = self.app.get(selfie_bgr)
        if not faces:
            raise FaceDetectionError(
                "Không phát hiện được khuôn mặt trong ảnh Selfie. Vui lòng chụp thẳng mặt, đủ sáng."
            )

        valid_faces = [
            f for f in faces if getattr(f, "det_score", 0.0) >= settings.FACE_DET_SCORE_THRESHOLD
        ]
        if not valid_faces:
            raise FaceDetectionError(
                f"Khuôn mặt trong ảnh Selfie không đủ rõ nét (độ tin cậy < {settings.FACE_DET_SCORE_THRESHOLD * 100:.0f}%)."
            )

        best_face = max(
            valid_faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        )

        normed_embedding = best_face.normed_embedding
        if normed_embedding is None or len(normed_embedding) == 0:
            raise FaceDetectionError("Không thể trích xuất vector khuôn mặt từ ảnh Selfie.")

        return normed_embedding.astype(float).tolist(), float(best_face.det_score)

    @staticmethod
    def compute_cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
        """
        Tính Cosine Similarity giữa 2 vector embedding đã chuẩn hóa L2.
        Giá trị trả về trong khoảng [-1.0, 1.0].
        """
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        sim = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(-1.0, min(1.0, sim))
