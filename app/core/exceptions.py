from fastapi import HTTPException, status


class CCCDVerificationException(HTTPException):
    """Lớp ngoại lệ cơ sở cho toàn bộ pipeline eKYC CCCD."""

    def __init__(self, detail: str, code: str = "VERIFICATION_FAILED"):
        super().__init__(
            status_code=422,
            detail={"code": code, "message": detail}
        )


class GeometricValidationError(CCCDVerificationException):
    """Lỗi vi phạm tiêu chuẩn hình học (thiếu góc, méo góc, quá gần/xa)."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="GEOMETRIC_VALIDATION_ERROR")


class ImageQualityError(CCCDVerificationException):
    """Lỗi chất lượng ảnh (mờ, nhòe, lóa flash, moiré chụp từ màn hình)."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="IMAGE_QUALITY_ERROR")


class QRCodeError(CCCDVerificationException):
    """Lỗi không đọc được hoặc dữ liệu mã QR không đúng chuẩn 7 trường."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="QR_CODE_ERROR")


class FaceDetectionError(CCCDVerificationException):
    """Lỗi không phát hiện được khuôn mặt chân dung hợp lệ trên thẻ."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="FACE_DETECTION_ERROR")


class CrossVerificationError(CCCDVerificationException):
    """Lỗi đối chiếu chéo giữa OCR và QR không khớp (nghi ngờ giả mạo)."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="CROSS_VERIFICATION_MISMATCH")


class SideMismatchError(CCCDVerificationException):
    """Lỗi sai mặt thẻ (ví dụ: yêu cầu mặt trước nhưng tải lên mặt sau hoặc ngược lại)."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, code="SIDE_MISMATCH")

