import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.core.exceptions import CCCDVerificationException
from app.schemas.cccd import CCCDVerificationResponse, HealthResponse
from app.services.cccd_processor import CCCDProcessor

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cccd_api")


_processor_instance: CCCDProcessor | None = None


def get_processor() -> CCCDProcessor:
    """Singleton getter đảm bảo CCCDProcessor luôn sẵn sàng cả trong lifespan và TestClient."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = CCCDProcessor()
    return _processor_instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Quản lý vòng đời ứng dụng FastAPI:
    Khởi tạo Singleton CCCDProcessor và tải sẵn toàn bộ model vào RAM trước khi nhận request.
    """
    logger.info("Đang khởi động CCCD eKYC Service & Tải các model Singleton...")
    processor = get_processor()
    app.state.processor = processor
    logger.info("CCCD eKYC Service đã sẵn sàng tiếp nhận request!")

    yield

    logger.info("Đang dừng dịch vụ và giải phóng tài nguyên...")
    processor.close()
    logger.info("Dịch vụ đã dừng hoàn tất.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend hiệu năng cao xác thực và trích xuất dữ liệu CCCD Việt Nam (eKYC) với mục tiêu dưới 60ms trên CPU.",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(CCCDVerificationException)
async def cccd_verification_exception_handler(
    request: Request, exc: CCCDVerificationException
) -> JSONResponse:
    """Xử lý ngoại lệ chuẩn hóa cho toàn bộ lỗi trong pipeline eKYC (HTTP 422)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.detail.get("code", "VERIFICATION_FAILED"),
            "message": exc.detail.get("message", "Xác thực thẻ CCCD thất bại."),
        },
    )


STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def root():
    """Giao diện Web Console trực quan để test & benchmark eKYC CCCD."""
    return FileResponse(STATIC_INDEX)


@app.get("/api/info", tags=["General"])
async def api_info():
    """Thông tin tổng quan về API."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "verify_endpoint": "/api/v1/cccd/verify",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Kiểm tra tình trạng hoạt động của hệ thống và các model Singleton."""
    processor = getattr(app.state, "processor", None) or get_processor()
    models_ready = processor is not None

    return HealthResponse(
        status="healthy" if models_ready else "degraded",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        models_loaded={
            "processor_initialized": models_ready,
            "wechat_qr": processor.qr_reader.wechat_detector is not None if processor else False,
            "insightface": processor.face_extractor.app is not None if processor else False,
            "rapid_ocr": processor.ocr_verifier.engine is not None if processor else False,
        },
    )


@app.post(
    "/api/v1/cccd/verify",
    response_model=CCCDVerificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["eKYC Verification"],
    summary="Xác thực và trích xuất dữ liệu thẻ CCCD Việt Nam",
)
async def verify_cccd(
    file: UploadFile = File(..., description="File ảnh thẻ CCCD định dạng JPEG, PNG, WEBP")
) -> CCCDVerificationResponse:
    """
    Pipeline eKYC Fail-Fast xử lý dưới 60ms trên CPU:
    1. Tiền xử lý, resize 640px, bắt 4 góc thẻ, validate hình học.
    2. Nắn phẳng thẻ về chuẩn cố định 1000 x 630 (Perspective Warp).
    3. Kiểm tra chất lượng: Chống mờ nét, lóa flash, và quét vân màn hình Moiré.
    4. Xử lý song song ROI: Đọc QR (WeChatQR/zxing) + Cắt ảnh 3x4 & Vector 512D (InsightFace), tự xoay 180°.
    5. OCR Số CCCD và Họ tên (RapidOCR) & Đối chiếu chéo chống sửa ảnh.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File tải lên phải là định dạng hình ảnh (image/jpeg, image/png, image/webp).",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dữ liệu file tải lên bị rỗng.",
        )

    processor: CCCDProcessor = getattr(app.state, "processor", None) or get_processor()
    return processor.process(image_bytes)
