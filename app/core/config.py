from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Cấu hình hệ thống eKYC CCCD Việt Nam.
    Bao gồm toàn bộ hằng số ngưỡng (thresholds) hình học, kiểm tra chất lượng,
    tọa độ ROI cắt thẻ và đường dẫn tới các model.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server settings
    APP_NAME: str = "CCCD Vietnam eKYC Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    THREAD_POOL_WORKERS: int = 4

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    WECHAT_QR_DIR: Path = MODELS_DIR / "wechat_qrcode"
    DOCALIGNER_MODEL_PATH: Path = MODELS_DIR / "docaligner-lcnet100.onnx"
    ASTERNET_MODEL_PATH: Path = MODELS_DIR / "asternet.onnx"
    ASTERNET_VOCAB_PATH: Path = MODELS_DIR / "asternet_vocab.json"
    INSIGHTFACE_NAME: str = "buffalo_sc"

    # 1. Kích thước & Tiền xử lý nắn phẳng thẻ
    CARD_WARP_WIDTH: int = 1000
    CARD_WARP_HEIGHT: int = 630
    CORNER_DETECT_RESIZE_MAX: int = 640

    # 2. Ngưỡng kiểm tra hình học (Geometric Validation)
    MIN_CORNER_BORDER_DISTANCE: int = 4   # px, tối thiểu cách mép ảnh
    MIN_CARD_AREA_RATIO: float = 0.05     # Cho phép ảnh chụp xa (thẻ >= 5% diện tích ảnh) để tự cắt khung chuẩn
    MAX_CARD_AREA_RATIO: float = 0.95     # Diện tích thẻ / toàn ảnh tối đa
    MIN_CARD_ASPECT_RATIO: float = 1.30   # width / height tối thiểu
    MAX_CARD_ASPECT_RATIO: float = 1.90   # width / height tối đa (chuẩn ID-1: 1.586)

    # 3. Ngưỡng kiểm tra chất lượng & Chống giả mạo (Quality & Anti-Spoof)
    MIN_SHARPNESS_LAPLACIAN_VAR: float = 65.0  # Phương sai Laplacian chống mờ/nhòe (tinh chỉnh cho ảnh thực tế)
    MIN_BRIGHTNESS_MEAN: float = 45.0          # Độ sáng trung bình tối thiểu (tránh ảnh quá tối)
    MAX_BRIGHTNESS_MEAN: float = 235.0         # Độ sáng trung bình tối đa (tránh cháy sáng toàn thẻ)
    GLARE_V_THRESHOLD: int = 250               # HSV Value > 250 (điểm lóa trắng gắt)
    GLARE_S_THRESHOLD: int = 25                # HSV Saturation < 25
    MAX_GLARE_CARD_RATIO: float = 0.08         # Lóa toàn thẻ tối đa 8%
    MAX_GLARE_QR_RATIO: float = 0.05           # Lóa vùng QR tối đa 5%
    MAX_GLARE_PORTRAIT_RATIO: float = 0.06     # Lóa vùng chân dung tối đa 6%
    ENABLE_MOIRE_CHECK: bool = True            # Bật/tắt kiểm tra vân màn hình
    MOIRE_MAX_PEAK_RATIO: float = 55.0         # Ngưỡng tỉ lệ đỉnh phổ tần số cao

    # 4. Tọa độ ROI trên ảnh thẻ đã nắn phẳng 1000 x 630
    # Định dạng (y1, y2, x1, x2)
    ROI_QR: tuple[int, int, int, int] = (10, 290, 680, 990)
    ROI_PORTRAIT: tuple[int, int, int, int] = (130, 595, 25, 375)
    ROI_CCCD_NUMBER: tuple[int, int, int, int] = (245, 315, 365, 760)
    ROI_FULL_NAME: tuple[int, int, int, int] = (325, 395, 360, 930)

    # 5. Cắt ảnh chân dung (lấy toàn bộ chân dung 3x4 gồm đầu + vai), InsightFace & Face Matching
    FACE_DET_SCORE_THRESHOLD: float = 0.65
    FACE_MATCH_SIMILARITY_THRESHOLD: float = 0.40  # Ngưỡng Cosine Similarity InsightFace buffalo_sc
    FACE_CROP_WIDTH_SCALE: float = 2.05
    FACE_CROP_HEIGHT_SCALE: float = 2.65
    FACE_TOP_HEAD_MARGIN_RATIO: float = 0.36
    INSIGHTFACE_DET_SIZE: tuple[int, int] = (320, 320)

    # 6. Fallback Corner Detection
    USE_ONNX_CORNER_DETECTOR: bool = True  # Dùng DocAligner ONNX hoặc Contour Fallback


settings = Settings()
