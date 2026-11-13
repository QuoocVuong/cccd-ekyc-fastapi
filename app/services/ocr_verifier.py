import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
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
    text = text.replace("Đ", "D").replace("đ", "d")
    nfd_text = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd_text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def normalize_name(text: str) -> str:
    """
    Chuẩn hóa họ tên: bỏ dấu tiếng Việt, viết hoa, loại bỏ ký tự đặc biệt thừa,
    và chuẩn hóa khoảng trắng đơn.
    """
    s = strip_accents(text).upper()
    s = re.sub(r"[^A-Z\s]", " ", s)
    return " ".join(s.split())


class VietnameseTextRecognizer:
    """
    Bộ nhận diện văn bản tiếng Việt có dấu chuẩn NFC dựa trên AsterNet ONNX (227 ký tự).
    Chạy thuần trên ONNXRuntime CPUExecutionProvider.
    """

    def __init__(
        self,
        model_path: Path = settings.ASTERNET_MODEL_PATH,
        vocab_path: Path = settings.ASTERNET_VOCAB_PATH,
    ):
        self.session: Optional[ort.InferenceSession] = None
        self.chars: list[str] = []
        self._init_model(model_path, vocab_path)

    def _init_model(self, model_path: Path, vocab_path: Path) -> None:
        if not model_path.exists() or not vocab_path.exists():
            logger.warning(
                f"Không tìm thấy AsterNet ONNX ({model_path}) hoặc vocab ({vocab_path}). "
                "Sẽ fallback sang RapidOCR."
            )
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.chars = json.load(f)["chars"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Khởi tạo AsterNet Vietnamese OCR ONNX thành công (227 ký tự có dấu).")

    @property
    def is_ready(self) -> bool:
        return self.session is not None and len(self.chars) > 0

    def recognize(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện 1 dòng văn bản tiếng Việt có dấu từ ảnh crop BGR.
        Trả về (text_nfc, confidence).
        """
        if not self.is_ready or crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return "", 0.0

        # Resize giữ tỷ lệ về chiều cao chuẩn 32px
        new_w = max(32, int(round(w * 32.0 / h)))
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, 32), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        blob = np.transpose(arr, (2, 0, 1))[None, ...]

        logits = self.session.run(None, {self.input_name: blob})[0]  # [time, 1, 227]
        frame_probs = logits[:, 0, :]  # [time, 227]
        preds = np.argmax(frame_probs, axis=-1)
        max_log_probs = np.max(frame_probs, axis=-1)

        # CTC Greedy Decode
        decoded_chars = []
        confs = []
        prev = 0
        for idx, log_p in zip(preds, max_log_probs):
            if idx != 0 and idx != prev and idx < len(self.chars):
                decoded_chars.append(self.chars[idx])
                confs.append(float(np.exp(log_p)))
            prev = idx

        text = unicodedata.normalize("NFC", "".join(decoded_chars).strip())
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf


class OCRVerifier:
    """
    Module OCR tiếng Việt chuẩn dấu kết hợp:
    - RapidOCR (Detection tìm bounding boxes các dòng chữ trên thẻ / VNeID)
    - AsterNet Vietnamese ONNX (Recognition chuẩn 227 ký tự tiếng Việt có dấu)
    - RapidOCR Recognition fallback nếu chưa có AsterNet.
    """

    def __init__(self):
        self.engine: Optional[RapidOCR] = None
        self.vi_rec = VietnameseTextRecognizer()
        self._init_ocr()

    def _init_ocr(self) -> None:
        try:
            self.engine = RapidOCR(
                det_use_cuda=False,
                cls_use_cuda=False,
                rec_use_cuda=False,
            )
            logger.info("Khởi tạo RapidOCR thành công.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo RapidOCR: {e}")
            raise

    def recognize_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện văn bản trên ảnh cắt ROI.
        Ưu tiên AsterNet Vietnamese ONNX để đọc chuẩn tiếng Việt có dấu, fallback sang RapidOCR.
        """
        if self.vi_rec.is_ready:
            # Nếu ROI cao (có thể chứa nhiều dòng hoặc khoảng trắng thừa), thử detect box bên trong trước nếu cao > 80px
            text, conf = self.vi_rec.recognize(crop_bgr)
            if text:
                return text, conf

        if self.engine is None:
            return "", 0.0

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
        executor=None,
    ) -> OCRInfo:
        """
        Cắt 2 vùng ROI Số CCCD và Họ tên trên thẻ 1000 x 630, chạy OCR tiếng Việt có dấu
        và đối chiếu chéo với dữ liệu mã QR.
        """
        ny1, ny2, nx1, nx2 = settings.ROI_CCCD_NUMBER
        crop_number = warped_card[ny1:ny2, nx1:nx2]

        my1, my2, mx1, mx2 = settings.ROI_FULL_NAME
        crop_name = warped_card[my1:my2, mx1:mx2]

        if executor is not None:
            future_num = executor.submit(self.recognize_crop, crop_number)
            future_name = executor.submit(self.recognize_crop, crop_name)
            raw_num_text, conf_num = future_num.result()
            raw_name_text, conf_name = future_name.result()
        else:
            raw_num_text, conf_num = self.recognize_crop(crop_number)
            raw_name_text, conf_name = self.recognize_crop(crop_name)

        ocr_digits = "".join(re.findall(r"\d", raw_num_text))
        avg_conf = (conf_num + conf_name) / 2.0 if (conf_num > 0 or conf_name > 0) else 0.0

        # Đối chiếu Số CCCD (100% khớp 12 số)
        qr_cccd = qr_info.cccd_number.strip()
        cccd_match = ocr_digits == qr_cccd
        if not cccd_match:
            raise CrossVerificationError(
                f"Số CCCD nhận diện qua OCR ('{ocr_digits}') không trùng khớp 100% với dữ liệu mã QR ('{qr_cccd}'). "
                "Cảnh báo: Nghi ngờ phôi thẻ đã bị tẩy xóa, dán đè số hoặc làm giả!"
            )

        # Đối chiếu Họ và Tên (chuẩn hóa NFC, bỏ dấu tiếng Việt để tránh lỗi trầy xước mất dấu thanh)
        norm_ocr_name = normalize_name(raw_name_text)
        norm_qr_name = normalize_name(qr_info.full_name)

        compact_ocr = norm_ocr_name.replace(" ", "")
        compact_qr = norm_qr_name.replace(" ", "")
        seq_ratio = SequenceMatcher(None, compact_ocr, compact_qr).ratio() if compact_qr else 0.0

        ocr_words = set(norm_ocr_name.split())
        qr_words = set(norm_qr_name.split())
        common_words = ocr_words.intersection(qr_words)
        word_ratio = len(common_words) / len(qr_words) if qr_words else 0.0

        name_match = (
            norm_ocr_name == norm_qr_name
            or word_ratio >= 0.70
            or seq_ratio >= 0.70
            or (len(compact_ocr) >= 6 and compact_ocr in compact_qr)
        )

        if not name_match:
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

    def extract_full_card_ocr(self, image_bgr: np.ndarray) -> dict:
        """
        OCR cơ bản toàn bộ văn bản tiếng Việt có dấu trên mặt thẻ CCCD hoặc ảnh chụp màn hình VNeID:
        1. Dùng RapidOCR Text Detection (use_det=True, use_rec=False) để lấy tất cả các dòng chữ.
        2. Dùng AsterNet Vietnamese ONNX nhận diện từng dòng chuẩn tiếng Việt có dấu.
        3. Bóc tách cấu trúc các trường thông tin quan trọng (Số CCCD, Họ tên, Ngày sinh, Giới tính...).
        """
        if self.engine is None:
            raise RuntimeError("RapidOCR engine chưa sẵn sàng.")

        img_h, img_w = image_bgr.shape[:2]
        lines: list[dict] = []

        if self.vi_rec.is_ready:
            # Chỉ chạy Detection để lấy bounding boxes chính xác
            det_res, _ = self.engine(image_bgr, use_det=True, use_cls=False, use_rec=False)
            if det_res:
                # Sắp xếp boxes từ trên xuống dưới, trái sang phải
                sorted_boxes = sorted(det_res, key=lambda b: (min(pt[1] for pt in b), min(pt[0] for pt in b)))
                for box in sorted_boxes:
                    xs = [int(pt[0]) for pt in box]
                    ys = [int(pt[1]) for pt in box]
                    # Padding 3px để không cắt mất dấu mũ/dấu nặng của chữ tiếng Việt
                    x1 = max(0, min(xs) - 3)
                    y1 = max(0, min(ys) - 3)
                    x2 = min(img_w, max(xs) + 3)
                    y2 = min(img_h, max(ys) + 3)
                    crop = image_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    text, conf = self.vi_rec.recognize(crop)
                    if text:
                        lines.append({
                            "text": text,
                            "confidence": round(conf, 4),
                            "box": box,
                        })
        else:
            # Fallback nếu không có AsterNet
            results, _ = self.engine(image_bgr, use_det=True, use_cls=False, use_rec=True)
            if results:
                for item in results:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        box, text, score = item[0], str(item[1]).strip(), float(item[2])
                        if text:
                            lines.append({
                                "text": text,
                                "confidence": round(score, 4),
                                "box": box,
                            })

        all_text = "\n".join(l["text"] for l in lines)
        cccd_match = re.search(r"\b\d{12}\b", all_text)
        dates = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", all_text)
        dob = dates[0] if len(dates) >= 1 else None
        expiry_date = dates[1] if len(dates) >= 2 else None

        gender = None
        if re.search(r"\bNam\b", all_text, re.IGNORECASE):
            gender = "Nam"
        elif re.search(r"\bN[uữ]\b", all_text, re.IGNORECASE):
            gender = "Nữ"

        # Tìm họ tên từ dòng sau label "Họ và tên" / "Họ, chữ đệm và tên" hoặc dòng ngay sau số CCCD
        full_name = None
        for i, l in enumerate(lines):
            t_low = l["text"].lower()
            if ("họ và tên" in t_low or "chữ đệm và tên" in t_low or "full name" in t_low) and i + 1 < len(lines):
                cand = lines[i + 1]["text"].strip()
                if len(cand) >= 4 and not re.search(r"\d", cand):
                    full_name = cand
                    break
        if not full_name and cccd_match:
            num_str = cccd_match.group(0)
            for i, l in enumerate(lines):
                if num_str in l["text"] and i + 1 < len(lines):
                    cand = lines[i + 1]["text"].strip()
                    if len(cand) >= 4 and not re.search(r"\d", cand):
                        full_name = cand
                        break

        return {
            "cccd_number": cccd_match.group(0) if cccd_match else None,
            "full_name": full_name,
            "dob": dob,
            "gender": gender,
            "expiry_date": expiry_date,
            "raw_lines": [{"text": l["text"], "confidence": l["confidence"]} for l in lines],
        }
