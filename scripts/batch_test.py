#!/usr/bin/env python3
"""
Script chạy kiểm thử hàng loạt trên thư mục ảnh CCCD / Thẻ Căn cước thực tế.
Hỗ trợ:
- Tự động bắt 4 góc (kể cả ảnh chụp xa) và cắt phẳng chuẩn 1000 x 630 -> lưu vào output/warped/
- Trích xuất khuôn mặt chân dung 3x4 -> lưu vào output/portraits/
- Hỗ trợ cả CCCD gắn chip 2021 (có QR) và Thẻ Căn cước 2024 (không QR mặt trước)
- Báo cáo độ nét, độ sáng, % lóa flash và thời gian xử lý -> output/report.json

Cách dùng:
    python scripts/batch_test.py --input "C:\\Users\\vuong\\Downloads\\test" --output "./test_results"
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Thêm thư mục gốc vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.exceptions import CCCDVerificationException
from app.services.card_detector import validate_geometry, warp_card
from app.services.cccd_processor import CCCDProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def save_portrait_from_b64(b64_uri: str, dest_path: Path) -> None:
    if "," in b64_uri:
        _, b64_data = b64_uri.split(",", 1)
    else:
        b64_data = b64_uri
    dest_path.write_bytes(base64.b64decode(b64_data))


def run_batch(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[LỖI] Không tìm thấy thư mục đầu vào: {input_dir}")
        sys.exit(1)

    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"[CẢNH BÁO] Không có file ảnh nào trong: {input_dir}")
        return

    warped_dir = output_dir / "warped"
    portraits_dir = output_dir / "portraits"
    warped_dir.mkdir(parents=True, exist_ok=True)
    portraits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang khởi tạo CCCDProcessor cho {len(image_files)} ảnh...")
    processor = CCCDProcessor()

    results = []
    success_count = 0

    print("\n" + "=" * 80)
    print(f" KẾT QUẢ KIỂM THỬ THƯ MỤC: {input_dir}")
    print("=" * 80)

    for idx, img_path in enumerate(image_files, 1):
        raw_bytes = img_path.read_bytes()
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        entry = {"file": img_path.name}

        try:
            # Lưu ảnh đã cắt phẳng 1000x630
            if img is not None:
                corners = processor.detector.detect(img)
                if corners is not None:
                    try:
                        v_corners = validate_geometry(corners, img.shape[1], img.shape[0])
                        warped = warp_card(img, v_corners)
                        cv2.imwrite(str(warped_dir / f"{img_path.stem}_warped.jpg"), warped)
                    except CCCDVerificationException:
                        pass

            # Chạy pipeline đầy đủ với strict_qr=False để hỗ trợ cả 2 loại căn cước
            resp = processor.process(raw_bytes, strict_qr=False)
            success_count += 1

            # Lưu ảnh chân dung 3x4 trích xuất
            portrait_file = portraits_dir / f"{img_path.stem}_portrait.jpg"
            save_portrait_from_b64(resp.portrait_base64, portrait_file)

            cccd_num = resp.qr_info.cccd_number if resp.qr_info else resp.ocr_info.cccd_number
            full_name = resp.qr_info.full_name if resp.qr_info else resp.ocr_info.full_name
            qm = resp.quality_metrics or {}

            print(
                f"[{idx:02d}/{len(image_files):02d}] ✅ {img_path.name} | {resp.card_type} | "
                f"Số: {cccd_num} | Tên: {full_name} | "
                f"Nét: {qm.get('sharpness')} | Sáng: {qm.get('brightness')} | "
                f"Lóa: {qm.get('glare_card_pct')}% | {resp.processing_time_ms.total_ms:.1f} ms"
            )

            entry.update({
                "status": "success",
                "card_type": resp.card_type,
                "rotated_180": resp.rotated_180,
                "cccd_number": cccd_num,
                "full_name": full_name,
                "ocr_info": resp.ocr_info.model_dump(),
                "qr_info": resp.qr_info.model_dump() if resp.qr_info else None,
                "quality_metrics": qm,
                "processing_time_ms": resp.processing_time_ms.model_dump(),
                "portrait_saved": str(portrait_file),
            })

        except CCCDVerificationException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            print(
                f"[{idx:02d}/{len(image_files):02d}] ❌ {img_path.name} | "
                f"{detail.get('code', 'ERROR')}: {detail.get('message')}"
            )
            entry.update({
                "status": "error",
                "error_code": detail.get("code", "VERIFICATION_FAILED"),
                "message": detail.get("message", str(exc)),
            })
        except Exception as exc:
            print(f"[{idx:02d}/{len(image_files):02d}] 💥 {img_path.name} | EXCEPTION: {exc}")
            entry.update({"status": "error", "error_code": "INTERNAL_ERROR", "message": str(exc)})

        results.append(entry)

    processor.close()

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 80)
    print(f"Hoàn tất: {success_count}/{len(image_files)} ảnh đạt chuẩn.")
    print(f"- Ảnh thẻ cắt chuẩn 1000x630: {warped_dir}")
    print(f"- Ảnh khuôn mặt trích xuất:   {portraits_dir}")
    print(f"- Báo cáo JSON chi tiết:      {report_path}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test CCCD eKYC trên thư mục ảnh local.")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn thư mục chứa ảnh CCCD")
    parser.add_argument("--output", "-o", default="test_results", help="Thư mục lưu kết quả")
    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
