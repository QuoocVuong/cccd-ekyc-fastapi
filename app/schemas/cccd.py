from typing import Optional
from pydantic import BaseModel, Field


class QRInfo(BaseModel):
    """Thông tin trích xuất và giải mã từ mã QR trên thẻ CCCD."""
    cccd_number: str = Field(..., description="Số định danh cá nhân 12 số")
    old_cmnd: Optional[str] = Field(None, description="Số CMND 9 số cũ (nếu có)")
    full_name: str = Field(..., description="Họ và tên chủ thẻ")
    dob: str = Field(..., description="Ngày sinh định dạng DDMMYYYY")
    gender: str = Field(..., description="Giới tính (Nam/Nữ)")
    address: str = Field(..., description="Nơi thường trú")
    issue_date: str = Field(..., description="Ngày cấp thẻ định dạng DDMMYYYY")


class OCRInfo(BaseModel):
    """Kết quả nhận diện OCR trên các vùng ROI và kết quả đối chiếu chéo."""
    cccd_number: str = Field(..., description="Số CCCD nhận diện bằng RapidOCR")
    full_name: str = Field(..., description="Họ tên nhận diện bằng RapidOCR")
    cccd_match: bool = Field(..., description="Trùng khớp 100% số CCCD giữa OCR và QR")
    name_match: bool = Field(..., description="Trùng khớp họ tên không dấu giữa OCR và QR")
    confidence_score: float = Field(..., description="Điểm tin cậy trung bình của OCR")


class ProcessingTimeMS(BaseModel):
    """Thống kê chi tiết thời gian xử lý từng bước (milliseconds)."""
    total_ms: float = Field(..., description="Tổng thời gian xử lý pipeline")
    corner_detection_ms: float = Field(..., description="Bắt 4 góc thẻ và validate hình học")
    perspective_warp_ms: float = Field(..., description="Nắn phẳng thẻ về chuẩn 1000x630")
    quality_check_ms: float = Field(..., description="Kiểm tra mờ, lóa flash và moiré màn hình")
    qr_and_face_parallel_ms: float = Field(..., description="Xử lý song song luồng QR và Face")
    ocr_cross_verify_ms: float = Field(..., description="OCR vùng số/họ tên và đối chiếu chéo")


class MRZInfo(BaseModel):
    """Dữ liệu giải mã từ 3 dòng MRZ chuẩn ICAO TD1 trên mặt sau CCCD / Căn cước."""
    document_number: Optional[str] = Field(None, description="Số CCCD/ĐDCN trích xuất từ MRZ")
    date_of_birth: Optional[str] = Field(None, description="Ngày sinh (DD/MM/YYYY)")
    gender: Optional[str] = Field(None, description="Giới tính (Nam/Nữ)")
    expiry_date: Optional[str] = Field(None, description="Ngày hết hạn (DD/MM/YYYY)")
    nationality: Optional[str] = Field(None, description="Quốc tịch (VNM)")
    full_name_mrz: Optional[str] = Field(None, description="Họ tên không dấu theo chuẩn MRZ")
    raw_mrz: list[str] = Field(default_factory=list, description="3 dòng MRZ gốc đọc được")
    checksum_valid: bool = Field(False, description="Kết quả kiểm tra check-digit ICAO")


class BackSideInfo(BaseModel):
    """Thông tin trích xuất từ mặt sau thẻ CCCD / Căn cước."""
    identifying_features: Optional[str] = Field(None, description="Đặc điểm nhận dạng")
    issue_date: Optional[str] = Field(None, description="Ngày cấp (DD/MM/YYYY)")
    place_of_issue: Optional[str] = Field(None, description="Nơi cấp / Cơ quan cấp")
    raw_lines: list[str] = Field(default_factory=list, description="Các dòng văn bản đọc được trên mặt sau")


class CCCDVerificationResponse(BaseModel):
    """Schema phản hồi chuẩn doanh nghiệp quốc tế cho xác thực eKYC CCCD Việt Nam."""
    request_id: str = Field(..., description="Mã định danh duy nhất (UUID v4) cho mỗi request")
    timestamp: str = Field(..., description="Thời điểm xử lý chuẩn ISO-8601 UTC")
    api_version: str = Field("1.0.0", description="Phiên bản API")
    status: str = Field("success", description="Trạng thái xử lý")
    is_valid: bool = Field(True, description="Thẻ hợp lệ và vượt qua toàn bộ kiểm tra")
    card_type: str = Field("CCCD_2021", description="Loại thẻ: CCCD_2021, CAN_CUOC_2024, hoặc VNEID_SCREEN")
    card_side: str = Field("FRONT", description="Mặt thẻ nhận diện được: FRONT (mặt trước), BACK (mặt sau), BOTH (VNeID 2 mặt)")
    vneid_layout: Optional[str] = Field(None, description="Loại ảnh VNeID: SINGLE_SIDE (1 mặt) hoặc DUAL_SIDE (2 mặt)")
    rotated_180: bool = Field(False, description="Thẻ có bị xoay ngược 180 độ hay không")
    qr_info: Optional[QRInfo] = Field(None, description="Dữ liệu trích xuất từ QR (nếu có)")
    ocr_info: Optional[OCRInfo] = Field(None, description="Dữ liệu trích xuất và đối chiếu OCR")
    mrz_info: Optional[MRZInfo] = Field(None, description="Dữ liệu trích xuất từ MRZ mặt sau (nếu có)")
    back_info: Optional[BackSideInfo] = Field(None, description="Thông tin mặt sau thẻ (nếu có)")
    portrait_base64: Optional[str] = Field(None, description="Ảnh chân dung 3x4 (Base64 JPEG, có ở mặt trước/VNeID)")
    face_embedding: list[float] = Field(default_factory=list, description="Vector khuôn mặt 512 chiều (có ở mặt trước/VNeID)")
    quality_metrics: Optional[dict[str, float]] = Field(None, description="Các chỉ số đo độ nét, độ sáng, độ lóa")
    warnings: list[str] = Field(default_factory=list, description="Danh sách cảnh báo cần lưu ý (nếu có)")
    processing_time_ms: ProcessingTimeMS = Field(..., description="Đo lường hiệu năng từng công đoạn")


class HealthResponse(BaseModel):
    """Trạng thái hoạt động của hệ thống và các model Singleton."""
    status: str = "healthy"
    service: str = "CCCD Vietnam eKYC Backend"
    version: str = "1.0.0"
    models_loaded: dict[str, bool]


class FaceMatchResponse(BaseModel):
    """Kết quả so khớp khuôn mặt 1:1 giữa ảnh Selfie và ảnh chân dung trên CCCD."""
    status: str = "success"
    is_match: bool = Field(..., description="True nếu cùng một người (Cosine Similarity >= ngưỡng)")
    similarity: float = Field(..., description="Điểm tương đồng Cosine [-1.0 .. 1.0]")
    similarity_percent: float = Field(..., description="Tỷ lệ giống nhau quy đổi ra phần trăm (0..100%)")
    threshold: float = Field(..., description="Ngưỡng quyết định khớp khuôn mặt")
    selfie_det_score: float = Field(..., description="Độ tin cậy phát hiện mặt trong ảnh Selfie")
    processing_time_ms: float = Field(..., description="Thời gian xử lý so khớp (ms)")


class BasicOCRLine(BaseModel):
    text: str
    confidence: float


class BasicOCRResponse(BaseModel):
    """Kết quả OCR cơ bản toàn bộ mặt thẻ CCCD đã nắn phẳng."""
    status: str = "success"
    cccd_number: Optional[str] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    expiry_date: Optional[str] = None
    raw_lines: list[BasicOCRLine] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Thời gian xử lý OCR toàn thẻ (ms)")
