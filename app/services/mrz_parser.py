import logging
import re
from typing import Optional

from app.schemas.cccd import MRZInfo

logger = logging.getLogger(__name__)


def _icao_check_digit(data: str) -> int:
    """Tính check digit chuẩn ICAO 9303 (trọng số 7, 3, 1 lặp lại)."""
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(data):
        if "0" <= ch <= "9":
            val = int(ch)
        elif "A" <= ch <= "Z":
            val = ord(ch) - ord("A") + 10
        else:
            val = 0
        total += val * weights[i % 3]
    return total % 10


def _format_yymmdd(yymmdd: str, is_expiry: bool = False) -> Optional[str]:
    """Chuyển đổi YYMMDD sang DD/MM/YYYY."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy = int(yymmdd[0:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]
    if is_expiry:
        yyyy = 2000 + yy if yy <= 80 else 1900 + yy
    else:
        yyyy = 1900 + yy if yy >= 30 else 2000 + yy
    return f"{dd}/{mm}/{yyyy}"


def parse_mrz_lines(raw_lines: list[str]) -> Optional[MRZInfo]:
    """
    Phân tích 3 dòng MRZ chuẩn ICAO TD1 trên mặt sau CCCD / Căn cước Việt Nam:
    - Line 1: IDVNM... (chứa số CCCD 12 số)
    - Line 2: YYMMDD<check><M/F>YYMMDD<check>VNM...
    - Line 3: HO<<TEN<DEM<<<
    """
    if not raw_lines:
        return None

    cleaned: list[str] = []
    for line in raw_lines:
        s = re.sub(r"\s+", "", line.upper())
        # Chuẩn hóa lỗi OCR phổ biến với dấu '<'
        if "<<" in s or s.startswith("IDVNM") or s.startswith("1DVNM") or s.startswith("IDVN"):
            s = s.replace("1DVNM", "IDVNM")
            cleaned.append(s)

    if not cleaned:
        return None

    line1 = next((l for l in cleaned if l.startswith("IDVN")), None)
    line2 = next((l for l in cleaned if re.search(r"\d{6}\d[MF<]\d{6}", l)), None)
    line3 = None
    if line2 and cleaned.index(line2) + 1 < len(cleaned):
        cand3 = cleaned[cleaned.index(line2) + 1]
        if "<<" in cand3 and not any(c.isdigit() for c in cand3[:10]):
            line3 = cand3

    if line1 is None and line2 is None:
        return None

    doc_num = None
    if line1:
        # Chuẩn CCCD Việt Nam: IDVNM + 9 số cuối + 1 check digit + 12 số CCCD đầy đủ
        m_vn = re.search(r"IDVN[A-Z0-9]{10,11}(\d{12})", line1)
        if m_vn:
            doc_num = m_vn.group(1)
        else:
            m12 = re.search(r"(\d{12})<*", line1[5:])
            if m12:
                doc_num = m12.group(1)
            else:
                m9 = re.search(r"IDVNM(\d{9})", line1)
                if m9:
                    doc_num = m9.group(1)

    dob = None
    gender = None
    expiry_date = None
    nationality = "VNM"
    checksum_valid = False

    if line2:
        m2 = re.search(r"(\d{6})(\d)([MF<])(\d{6})(\d)([A-Z<]{3})?", line2)
        if m2:
            dob_raw, dob_chk, sex_ch, exp_raw, exp_chk, nat = m2.groups()
            dob = _format_yymmdd(dob_raw, is_expiry=False)
            expiry_date = _format_yymmdd(exp_raw, is_expiry=True)
            if sex_ch == "M":
                gender = "Nam"
            elif sex_ch == "F":
                gender = "Nữ"
            if nat and nat.replace("<", ""):
                nationality = nat.replace("<", "")

            try:
                chk1_ok = _icao_check_digit(dob_raw) == int(dob_chk)
                chk2_ok = _icao_check_digit(exp_raw) == int(exp_chk)
                checksum_valid = chk1_ok and chk2_ok
            except ValueError:
                checksum_valid = False

    full_name_mrz = None
    if line3:
        parts = [p.replace("<", " ").strip() for p in line3.split("<<") if p.replace("<", "").strip()]
        if parts:
            full_name_mrz = " ".join(parts)

    return MRZInfo(
        document_number=doc_num,
        date_of_birth=dob,
        gender=gender,
        expiry_date=expiry_date,
        nationality=nationality,
        full_name_mrz=full_name_mrz,
        raw_mrz=[l for l in (line1, line2, line3) if l],
        checksum_valid=checksum_valid,
    )
