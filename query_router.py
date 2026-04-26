# ── new: query_router.py ─────────────────────────────────────────
import re
from typing import Dict, Any, Tuple, Optional

PROGRAM_ALIASES = {
    r"\bIME\b": "IME_MS",
    r"\b석사\b": "MS",
    r"\b박사\b": "PHD",
    r"\b학부\b": "UG",
}

def _norm_program(text: str) -> Optional[str]:
    for pat, norm in PROGRAM_ALIASES.items():
        if re.search(pat, text, flags=re.I):
            return norm
    return None

def _norm_cohort(text: str) -> Optional[str]:
    # 2023학번 / 23학번 / (20)23 등에서 4자리 연도 추출
    m = re.search(r"(20\d{2})\s*학?번?", text)
    if m:
        return f"Cohort_{m.group(1)}"
    # 백업: 2자리 연도
    m2 = re.search(r"\b(\d{2})\s*학?번?\b", text)
    if m2:
        return f"Cohort_20{int(m2.group(1)):02d}"
    return None

def _int(m):
    try: return int(m)
    except: return None

def query_router(user_input: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    returns:
      meta_filter:   FAISS 메타필터(dict)
      routing_hints: 답변/랭킹용 힌트(uri/article/clause/latest/table 등)
    """
    q = user_input or ""
    meta: Dict[str, Any] = {}

    # 제15조, 제15조 2항, 15조 2항 등
    m_art = re.search(r"제?\s*(\d{1,3})\s*조", q)
    if m_art:
        meta["articleNumber"] = _int(m_art.group(1))
    m_cls = re.search(r"(?:제?\s*\d{1,3}\s*조)?\s*(\d{1,2})\s*항", q)
    if m_cls:
        meta["clauseNumber"] = _int(m_cls.group(1))

    # 표/테이블 요청 힌트
    wants_table = bool(re.search(r"\b(표|table)\b", q, flags=re.I))
    if wants_table:
        meta["contentType"] = "table"

    # 페이지 직접 지목 (p.12, 12페이지)
    m_pg = re.search(r"(?:p\.|페이지)\s*([0-9]{1,4})", q, flags=re.I)
    if m_pg:
        meta["page"] = _int(m_pg.group(1))

    # 프로그램/코호트 정규화
    prog = _norm_program(q)
    if prog: meta["program"] = prog
    coh  = _norm_cohort(q)
    if coh:  meta["cohort"]  = coh

    # 신규(소융 데모): 트랙/전공구분/학수번호
    m_track = re.search(r"(게임콘텐츠|데이터사이언스|로봇비전|융합리더|GameContents|DataScience|RobotVision|ConvergenceLeader)", q, flags=re.I)
    if m_track:
        track_map = {
            "게임콘텐츠": "GameContents", "데이터사이언스": "DataScience",
            "로봇비전": "RobotVision", "융합리더": "ConvergenceLeader",
            "gamecontents": "GameContents", "datascience": "DataScience",
            "robotvision": "RobotVision", "convergenceleader": "ConvergenceLeader",
        }
        key = m_track.group(1)
        meta["track"] = track_map.get(key, track_map.get(key.lower(), key))

    if re.search(r"단일전공", q):
        meta["degreeType"] = "single"
    elif re.search(r"(다전공|복수전공)", q):
        meta["degreeType"] = "multi"
    elif re.search(r"부전공", q):
        meta["degreeType"] = "minor"

    m_sc = re.search(r"\b([A-Z]{2,5}\d{3,4})\b", user_input or "")
    if m_sc:
        meta["subjectCode"] = m_sc.group(1).upper()

    # 라우팅 힌트(후처리 랭킹에 사용)
    hints = {
        "wants_table": wants_table,
        "articleNumber": meta.get("articleNumber"),
        "clauseNumber": meta.get("clauseNumber"),
        "program": meta.get("program"),
        "cohort": meta.get("cohort"),
        "track": meta.get("track"),
        "degreeType": meta.get("degreeType"),
        "subjectCode": meta.get("subjectCode"),
    }
    return meta, hints
