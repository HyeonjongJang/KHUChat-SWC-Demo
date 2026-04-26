# utils.py
# 공통 유틸:
#  - JSONL 저장/로드
#  - 메타데이터 정규화(스키마/프로그램/코호트/조·항/콘텐츠 타입)
#  - URN/HTTP 영구 URI 생성
#  - sourceFile/md5/페이지 정규화
#
# 사용처 예:
#   from utils import attach_uri_and_schema, save_docs_to_jsonl, load_docs_from_jsonl
#
# 주의: 외부 의존성 없이 표준 라이브러리만 사용

from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Iterable, Optional, Tuple, Dict, Any

# LangChain Document 호환 (langchain==0.3 계열 지원)
try:
    from langchain.schema import Document as LCDocument  # type: ignore
except Exception:
    try:
        from langchain_core.documents import Document as LCDocument  # type: ignore
    except Exception:
        LCDocument = dict  # 완전 폴백(저장 전용)

# ─────────────────────────────────────────────────────────────
# 기본 설정/정규화 규칙
# ─────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"

PROGRAM_SET = {
    "UG", "MS", "PHD", "IME_MS", "IME_PHD",  # 필요한 경우 확장
}

HTTP_URI_BASE = "https://kg.khu.ac.kr/reg"  # 영구 HTTP URI 네임스페이스
ALLOWED_CONTENT_TYPES = {"text", "table", "annex", "appendix"}

_ARTICLE_RE = re.compile(r"(\d+)")  # "제15조" 등에서 숫자만 뽑기


# ─────────────────────────────────────────────────────────────
# JSONL 저장/로드
# ─────────────────────────────────────────────────────────────
def save_docs_to_jsonl(docs: Iterable[LCDocument], jsonl_path: str) -> None:
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as jsonl_file:
        for doc in docs:
            # LangChain Document(Pydantic) 우선 사용
            try:
                s = doc.json(ensure_ascii=False)  # type: ignore[attr-defined]
            except Exception:
                # 예비 경로: dict/model_dump 가능성
                to_dict = getattr(doc, "dict", None) or getattr(doc, "model_dump", None)
                if callable(to_dict):
                    s = json.dumps(to_dict(), ensure_ascii=False)
                else:
                    s = json.dumps(
                        {
                            "page_content": getattr(doc, "page_content", str(doc)),
                            "metadata": getattr(doc, "metadata", {}),
                        },
                        ensure_ascii=False,
                    )
            jsonl_file.write(s + "\n")


def load_docs_from_jsonl(jsonl_path: str):
    items = []
    if not os.path.exists(jsonl_path):
        return items
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ─────────────────────────────────────────────────────────────
# 메타 정규화/URI 유틸
# ─────────────────────────────────────────────────────────────
def compute_md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def normalize_program(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    x = str(v).strip().upper().replace("-", "_")
    return x if x in PROGRAM_SET else None


def normalize_cohort(v: Optional[str]) -> Optional[str]:
    """
    "2023" → "Cohort_2023" 형식 강제. 4자리 연도만 허용.
    """
    if not v:
        return None
    s = "".join(ch for ch in str(v) if ch.isdigit())
    if len(s) == 4 and s.startswith("20"):
        return f"Cohort_{s}"
    return None


def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def coerce_article_clause(md: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    JSON 메타에 article_number / articleNumber / "제N조" 형태가 섞여 있어도 흡수.
    clause는 없으면 None
    """
    a = md.get("articleNumber") or md.get("article_number") or md.get("articleNo") or md.get("article")
    if isinstance(a, str):
        m = _ARTICLE_RE.search(a)
        a = m.group(1) if m else a
    a = _to_int(a)

    c = md.get("clauseNumber") or md.get("clause_no") or md.get("clause")
    if isinstance(c, str):
        m = _ARTICLE_RE.search(c)
        c = m.group(1) if m else c
    c = _to_int(c)

    return a, c


def infer_content_type(meta: Dict[str, Any], page_content: str) -> str:
    """
    표/텍스트 구분: 메타의 content_type/contentType 우선,
    없으면 텍스트 내 마크다운 테이블 패턴으로 추정.
    """
    ct = (meta.get("contentType") or meta.get("content_type") or "").strip().lower()
    if ct in ALLOWED_CONTENT_TYPES:
        return "table" if ct == "table" else ct or "text"

    # 간단 휴리스틱: 파이프(|) 테이블 감지 → table
    if page_content and page_content.count("|") >= 4 and "\n| ---" in page_content:
        return "table"
    return "text"


def build_http_uris(code: str, vdate: str, art: Optional[int], cl: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """
    영구 HTTP URI 생성.
    예: https://kg.khu.ac.kr/reg/AA-2024-09-01#art15 / #art15-cl2
    """
    if not (code and vdate and art is not None):
        return None, None
    base = f"{HTTP_URI_BASE}/{code}-{vdate}"
    article_uri = f"{base}#art{art}"
    clause_uri = f"{article_uri}-cl{cl}" if cl is not None else None
    return article_uri, clause_uri


def make_urn(code: str, vdate: str, art: Optional[int], cl: Optional[int]) -> Optional[str]:
    """
    URN 규칙: urn:khu:reg:{code}:{versionDate}:art{N}[:cl{M}]
    """
    if not (code and vdate and art is not None):
        return None
    cl_suffix = f":cl{cl}" if cl is not None else ""
    return f"urn:khu:reg:{code}:{vdate}:art{art}{cl_suffix}"


def attach_uri_and_schema(meta: Dict[str, Any], page_content: str) -> Dict[str, Any]:
    """
    한 레코드의 메타를 표준화하고, URN/HTTP URI, sourceFile, md5 등을 보강한다.
    (파이프라인 어디서든 재사용 가능)
    """
    m = dict(meta or {})

    # 0) 출처/재현성
    if m.get("sourceFile") is None:
        m["sourceFile"] = m.get("filename") or None
    m["md5"] = compute_md5_text(page_content)

    # 1) 스키마 필수 라인업
    m.setdefault("schema_version", SCHEMA_VERSION)
    m.setdefault("documentCode", (m.get("document_code") or m.get("code") or "").strip())
    m.setdefault("versionDate", (m.get("versionDate") or m.get("version_date") or "").strip() or None)

    # 기간 키(없으면 존재만 보장)
    ef = m.get("effectiveFrom") or m.get("effective_from") or None
    eu = m.get("effectiveUntil") or m.get("effective_until") or None
    m["effectiveFrom"] = ef or None
    m["effectiveUntil"] = eu or None

    # 2) 페이지 정규화
    page = m.get("page") or m.get("page_number") or m.get("pageNumber")
    if page is not None:
        try:
            m["page"] = int(page)
        except Exception:
            m["page"] = page  # 불가피할 때 원문 유지

    # 3) program/cohort 정규형
    m["program"] = normalize_program(m.get("program"))
    m["cohort"] = normalize_cohort(m.get("cohort") or m.get("year") or m.get("student_year"))

    # 4) contentType 정규화
    m["contentType"] = infer_content_type(m, page_content)

    # 5) article/clause 정규화
    a, c = coerce_article_clause(m)
    if a is not None:
        m["articleNumber"] = a
    if "clauseNumber" not in m:
        m["clauseNumber"] = c  # None 가능

    # 6) 관계 후보 필드 보장
    for k in ("overrides", "cites", "hasExceptionFor"):
        if k not in m or m[k] is None:
            m[k] = []

    # 7) URN + HTTP 영구 URI 동시 부여
    code = (m.get("documentCode") or "").strip()
    vdate = (m.get("versionDate") or "").strip()
    art = m.get("articleNumber")
    cl = m.get("clauseNumber")

    m["uri"] = make_urn(code, vdate, art, cl)
    article_http, clause_http = build_http_uris(code, vdate, art, cl)
    m["articleUri"] = article_http
    m["clauseUri"] = clause_http

    return m
