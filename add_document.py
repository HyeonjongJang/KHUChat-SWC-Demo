# add_document.py  — JSON/PDF 청크 수집 + 메타데이터 표준화 + 오버라이드 주입 + 인덱싱/병합
# Phase 1: 메타데이터 표준화(URI/스키마/정규화) + HTTP URI/소스/MD5 보강 + CLI 오버라이드 지원

from __future__ import annotations

import os, re, shutil, argparse, datetime, math, json, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import PDFMinerLoader, NotebookLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.schema import Document as LCDocument
except Exception:
    from langchain_core.documents import Document as LCDocument  # fallback

# 프로젝트 유틸(JSONL 저장/로드)
from utils import load_docs_from_jsonl, save_docs_to_jsonl

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 상수/정의
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"

PROGRAM_SET = {"UG", "MS", "PHD", "IME_MS", "IME_PHD"}

# 신규(소융 데모): 트랙·전공구분·학수번호용 화이트리스트/패턴
TRACK_SET = {"GameContents", "DataScience", "RobotVision", "ConvergenceLeader"}
DEGREE_TYPE_SET = {"single", "multi", "minor"}
SUBJECT_CODE_RE = re.compile(r"^[A-Z]{2,5}\d{3,4}$")

# 기본 HTTP 네임스페이스 (CLI --http-base로 바꿀 수 있음)
DEFAULT_HTTP_URI_BASE = "https://kg.khu.ac.kr/reg"

CATEGORIES = {
    "regulations":          "규정",
    "undergrad_rules":      "학부 시행세칙",
    "grad_rules":           "대학원 시행세칙",
    "academic_system":      "학사제도",
    "software_convergence": "소프트웨어융합학과",
    "khu_rules_2026":       "2026 학사운영 규정집",
    "khu_guide_2026":       "2026 학사제도 안내",
    "khu_2026":             "2026 학사 종합 (규정집+안내)",
}

BASE        = Path(".")
FAISS_BASE  = BASE / "faiss_db"
TODO_BASE   = BASE / "todo_documents"
PAST_BASE   = BASE / "past_documents"
DOCS_BASE   = BASE / "docs"
BACKUP_BASE = BASE / "backup"

SUPPORTED_EXTS = {".pdf", ".txt", ".ipynb", ".json", ".jsonl"}


@dataclass
class Overrides:
    document_code: Optional[str] = None  # e.g., RS
    version_date: Optional[str] = None   # e.g., 2024-09-01
    program: Optional[str] = None        # e.g., UG
    cohort: Optional[str] = None         # e.g., 2022 or Cohort_2022
    track: Optional[str] = None          # 신규: GameContents 등
    degree_type: Optional[str] = None    # 신규: single|multi|minor
    http_base: str = DEFAULT_HTTP_URI_BASE


# 전역 컨텍스트(각 함수에서 접근)
CTX = Overrides()


# ─────────────────────────────────────────────────────────────────────────────
# 정규화/스키마 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _norm_spaces(s: str) -> str:
    # \x00: pdfplumber가 한글 PDF에서 단어 경계로 NULL을 넣는 경우가 있음
    # \x0c: form feed
    s = (s or "").replace("\x00", " ").replace("\x0c", " ").replace("\n", " ")
    return re.sub(r"\s{2,}", " ", s).strip()

def _make_source_prefix(filename: str) -> str:
    return f"Source : {filename}\n" if filename else ""

def _norm_program(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    x = str(v).strip().upper().replace("-", "_")
    return x if x in PROGRAM_SET else None

def _norm_cohort(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    # "2022" or "Cohort_2022" → Cohort_2022
    s = "".join(ch for ch in str(v) if ch.isdigit())
    if len(s) == 4 and s.startswith("20"):
        return f"Cohort_{s}"
    # 이미 Cohort_YYYY 라면 그대로 둘 수도 있지만, 일관성을 위해 변환만 허용
    m = re.fullmatch(r"Cohort_(20\d{2})", str(v))
    if m:
        return f"Cohort_{m.group(1)}"
    return None

# ---- 신규(소융 데모): track / degreeType / subjectCode / annexNumber ----
TRACK_ALIAS_KO = {
    "게임콘텐츠": "GameContents",
    "게임": "GameContents",
    "데이터사이언스": "DataScience",
    "데이터": "DataScience",
    "로봇비전": "RobotVision",
    "로봇": "RobotVision",
    "비전": "RobotVision",
    "융합리더": "ConvergenceLeader",
    "융합": "ConvergenceLeader",
}

def _norm_track(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip()
    if s in TRACK_SET:
        return s
    s_low = s.lower()
    for t in TRACK_SET:
        if t.lower() == s_low:
            return t
    for k, norm in TRACK_ALIAS_KO.items():
        if k in s:
            return norm
    return None

DEGREE_TYPE_ALIAS = {
    "단일전공": "single", "단일": "single",
    "다전공": "multi", "복수전공": "multi", "복수": "multi",
    "부전공": "minor",
}

def _norm_degree_type(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip()
    if s.lower() in DEGREE_TYPE_SET:
        return s.lower()
    for k, norm in DEGREE_TYPE_ALIAS.items():
        if k in s:
            return norm
    return None

def _norm_subject_code(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip().upper().replace(" ", "")
    return s if SUBJECT_CODE_RE.match(s) else None

def _norm_annex_number(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        digits = re.sub(r"\D", "", str(v))
        n = int(digits) if digits else 0
        return n if n > 0 else None
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _parse_article_clause(md: dict) -> Tuple[Optional[int], Optional[int]]:
    a = md.get("articleNumber") or md.get("article_number") or md.get("articleNo") or md.get("article")
    if isinstance(a, str):
        m = re.search(r"(\d+)", a)
        a = m.group(1) if m else a
    a = _to_int(a)

    c = md.get("clauseNumber") or md.get("clause_no") or md.get("clause")
    if isinstance(c, str):
        m = re.search(r"(\d+)", c)
        c = m.group(1) if m else c
    c = _to_int(c)

    return a, c

def _infer_content_type(md: dict, page_content: str) -> str:
    ct = (md.get("content_type") or md.get("contentType") or "").strip().lower()
    if ct == "table":
        return "table"
    # 간단한 마크다운 테이블 감지
    if page_content and page_content.count("|") >= 4 and "\n| ---" in page_content:
        return "table"
    return "text"

def _build_http_uris(code: str, vdate: str, art: Optional[int], cl: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    if not (code and vdate and (art is not None)):
        return None, None
    base = f"{CTX.http_base.rstrip('/')}/{code}-{vdate}"
    article_uri = f"{base}#art{art}"
    clause_uri = f"{article_uri}-cl{cl}" if cl is not None else None
    return article_uri, clause_uri

def _compute_md5_from_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def _attach_uri_and_schema(meta: dict, page_content: str) -> dict:
    """
    스키마 필수 필드 보강 + 오버라이드 적용 + URN/HTTP URI 동시 부여
    """
    m = dict(meta or {})

    # 0) 소스/지문
    if m.get("sourceFile") is None:
        m["sourceFile"] = m.get("filename") or None
    m["md5"] = _compute_md5_from_text(page_content)

    # 1) 스키마 기본
    m.setdefault("schema_version", SCHEMA_VERSION)

    # 2) documentCode / versionDate 오버라이드 적용
    if CTX.document_code:
        m["documentCode"] = CTX.document_code.strip()
    else:
        m.setdefault("documentCode", (m.get("document_code") or m.get("code") or "").strip())

    if CTX.version_date:
        m["versionDate"] = CTX.version_date.strip()
    else:
        m.setdefault("versionDate", (m.get("versionDate") or m.get("version_date") or "").strip() or None)

    # 3) 기간(없으면 None)
    ef = m.get("effectiveFrom") or m.get("effective_from")
    eu = m.get("effectiveUntil") or m.get("effective_until")
    m["effectiveFrom"] = ef or None
    m["effectiveUntil"] = eu or None

    # 4) 페이지
    page = m.get("page") or m.get("page_number") or m.get("pageNumber")
    if page is not None:
        try:
            m["page"] = int(page)
        except Exception:
            m["page"] = page

    # 5) program/cohort 우선순위: 청크 메타 > CLI 오버라이드 > 기타 키
    #    (학번 분기 부칙·경과조치 청크가 자체적으로 cohort을 박았으면 보존해야 함)
    program = m.get("program") or CTX.program
    cohort  = m.get("cohort") or CTX.cohort or m.get("year") or m.get("student_year")
    m["program"] = _norm_program(program)
    m["cohort"]  = _norm_cohort(cohort)

    # 5.5) 신규(소융 데모): track / degreeType / subjectCode / annexNumber 정규화
    #      track/degreeType 도 청크 메타 우선
    m["track"]       = _norm_track(m.get("track") or CTX.track)
    m["degreeType"]  = _norm_degree_type(m.get("degreeType") or m.get("degree_type") or CTX.degree_type)
    m["subjectCode"] = _norm_subject_code(m.get("subjectCode") or m.get("subject_code") or m.get("courseCode"))
    m["annexNumber"] = _norm_annex_number(m.get("annexNumber") or m.get("annex_number") or m.get("annex"))

    # 6) contentType
    m["contentType"] = _infer_content_type(m, page_content)

    # 7) article/clause 정규화
    a, c = _parse_article_clause(m)
    if a is not None:
        m["articleNumber"] = a
    if c is not None:
        m["clauseNumber"] = c
    else:
        m.setdefault("clauseNumber", None)

    # 8) 관계 필드 존재 보장
    for k in ("overrides", "cites", "hasExceptionFor"):
        if k not in m or m[k] is None:
            m[k] = []

    # 9) URI들
    code  = (m.get("documentCode") or "").strip()
    vdate = (m.get("versionDate") or "").strip()
    art   = m.get("articleNumber")
    cl    = m.get("clauseNumber")

    # URN
    if code and vdate and (art is not None):
        cl_suffix = f":cl{cl}" if (cl is not None) else ""
        m["uri"] = f"urn:khu:reg:{code}:{vdate}:art{art}{cl_suffix}"
    else:
        m.setdefault("uri", None)

    # HTTP
    article_http, clause_http = _build_http_uris(code, vdate, art, cl)
    m["articleUri"] = article_http
    m["clauseUri"]  = clause_http

    return m


# ─────────────────────────────────────────────────────────────────────────────
# 로더
# ─────────────────────────────────────────────────────────────────────────────
def _as_document(page_content: str, metadata: Optional[dict] = None) -> LCDocument:
    return LCDocument(page_content=page_content, metadata=metadata or {})

def _load_pdf_txt_ipynb(path: Path) -> List[LCDocument]:
    if path.suffix.lower() == ".txt":
        docs = TextLoader(str(path)).load()
    elif path.suffix.lower() == ".pdf":
        docs = PDFMinerLoader(str(path)).load()
        for d in docs:
            d.page_content = _norm_spaces(d.page_content)
    elif path.suffix.lower() == ".ipynb":
        docs = NotebookLoader(str(path), include_outputs=False, remove_newline=True).load()
    else:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
    splits = splitter.split_documents(docs)

    for d in splits:
        meta = dict(d.metadata or {})
        meta["filename"] = path.name
        d.page_content = _make_source_prefix(path.name) + (d.page_content or "")
        meta = _attach_uri_and_schema(meta, d.page_content)
        d.metadata = meta
    return splits

def _coerce_json_obj_to_doc(obj: dict, default_fname: str) -> Optional[LCDocument]:
    if not isinstance(obj, dict):
        return None
    text = obj.get("text") or obj.get("page_content") or ""
    md   = obj.get("metadata") or obj.get("meta") or {}
    if not isinstance(md, dict):
        md = {}

    doc_title = (md.get("document_title") or "").strip()
    filename  = md.get("filename") or (f"{doc_title}.pdf" if doc_title else default_fname)
    page_content = _make_source_prefix(filename) + _norm_spaces(str(text))

    meta = dict(md)
    meta["filename"] = filename
    meta = _attach_uri_and_schema(meta, page_content)
    return _as_document(page_content, meta)

def _load_json_chunk(path: Path) -> List[LCDocument]:
    docs: List[LCDocument] = []
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return docs

    # JSONL 스타일? (.jsonl 확장자면 무조건 JSONL, 그 외엔 휴리스틱)
    is_jsonl = path.suffix.lower() == ".jsonl"
    if is_jsonl or ("\n" in raw and not raw.lstrip().startswith(("{", "["))):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                d = _coerce_json_obj_to_doc(obj, default_fname=path.name)
                if d: docs.append(d)
            except Exception:
                continue
        return docs

    # JSON (obj or array)
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        return docs

    if isinstance(data, dict) and ("text" in data or "page_content" in data):
        d = _coerce_json_obj_to_doc(data, default_fname=path.name)
        if d: docs.append(d)
        return docs

    if isinstance(data, list):
        for obj in data:
            d = _coerce_json_obj_to_doc(obj, default_fname=path.name)
            if d: docs.append(d)
    return docs

def _load_path_as_documents(path: Path) -> List[LCDocument]:
    ext = path.suffix.lower()
    if ext in {".pdf", ".txt", ".ipynb"}:
        return _load_pdf_txt_ipynb(path)
    if ext in {".json", ".jsonl"}:
        return _load_json_chunk(path)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 수집/임베딩
# ─────────────────────────────────────────────────────────────────────────────
def _gather_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

def _build_index_in_batches(splits: List[LCDocument], emb, docs_per_batch: int = 32) -> Optional[FAISS]:
    total = len(splits)
    if total == 0:
        return None
    vs_local = None
    num_batches = math.ceil(total / docs_per_batch)
    for bi in range(num_batches):
        start, end = bi * docs_per_batch, min((bi + 1) * docs_per_batch, total)
        batch = splits[start:end]
        print(f"   → 인덱스 배치 {bi+1}/{num_batches} (문서 {len(batch)}개)")
        if vs_local is None:
            vs_local = FAISS.from_documents(batch, embedding=emb)
        else:
            vs_local.add_documents(batch)
    return vs_local

def _process_category(category_slug: str, cohort: Optional[str], source_dir: Optional[Path]) -> Tuple[List[LCDocument], Optional[FAISS]]:
    """
    입력 루트 결정:
      - source_dir가 주어지면 그 경로에서 직접 수집
      - 아니면 기본 todo_documents/<category>[/<cohort>]
    """
    emb = OpenAIEmbeddings(model="text-embedding-3-large")

    if source_dir:
        root = source_dir
        label_suffix = f"(source={source_dir})"
    else:
        root = TODO_BASE / category_slug / cohort if cohort else TODO_BASE / category_slug
        label_suffix = f"(todo_documents/{category_slug}{'/' + cohort if cohort else ''})"

    past_dir = PAST_BASE / category_slug / cohort if cohort else PAST_BASE / category_slug
    past_dir.mkdir(parents=True, exist_ok=True)

    files = _gather_files(root)
    if not files:
        label = f"{CATEGORIES[category_slug]} | {category_slug}" + (f" | cohort={cohort}" if cohort else "")
        print(f"[{label}] 처리할 파일이 없습니다: {label_suffix}")
        return [], None

    all_splits: List[LCDocument] = []
    for i, f in enumerate(sorted(files), 1):
        rel = f.relative_to(root) if str(f).startswith(str(root)) else f.name
        print(f"[{i}/{len(files)}] 로딩/분할: {rel}")
        try:
            docs = _load_path_as_documents(f)
            # 카테고리/코호트 메타 주입 + 오버라이드 재보정
            for d in docs:
                meta = dict(d.metadata or {})
                meta["category"] = category_slug
                # 청크에 이미 cohort가 박혀 있으면(예: 학번 분기 부칙·경과조치) 보존
                if cohort and not meta.get("cohort"):
                    meta["cohort"] = cohort
                meta = _attach_uri_and_schema(meta, d.page_content)
                d.metadata = meta
            if docs:
                print(f"   → 청크 수: {len(docs)}")
                all_splits.extend(docs)
            else:
                print("   → 건너뜀(로더가 문서를 만들지 못함)")
        finally:
            # source_dir로 주어진 경우에도 past_documents로 이동(중복 인덱싱 방지)
            try:
                target = past_dir / f.name
                if f.resolve() != target.resolve():
                    shutil.move(str(f), str(target))
            except Exception as e:
                print(f"   → 이동 실패: {e}")

    if not all_splits:
        print(f"[{CATEGORIES[category_slug]}] 생성된 청크가 없습니다. 인덱스를 저장하지 않습니다.")
        return [], None

    vs_new = _build_index_in_batches(all_splits, emb, docs_per_batch=32)
    return all_splits, vs_new


# ─────────────────────────────────────────────────────────────────────────────
# 병합/저장
# ─────────────────────────────────────────────────────────────────────────────
def _merge_and_save(category_slug: str, docs: List[LCDocument], vectorstore: Optional[FAISS], cohort: Optional[str] = None):
    """
    - 저장: faiss_db/<category>[/<cohort>]/{index.faiss,index.pkl}
    - 문서 메타: docs/<category>[/<cohort>]/doc.jsonl (기존과 합쳐 백업)
    - 백업: backup/<category>[/<cohort or all>]/<timestamp>/
    """
    faiss_dir = FAISS_BASE / category_slug / cohort if cohort else FAISS_BASE / category_slug
    faiss_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    index_faiss = faiss_dir / "index.faiss"
    index_pkl   = faiss_dir / "index.pkl"

    # 기존 인덱스 → 병합 후 백업
    if index_faiss.exists() and index_pkl.exists():
        print("기존 인덱스 발견 → 병합 후 백업")
        emb = OpenAIEmbeddings(model="text-embedding-3-large")
        past_vs = FAISS.load_local(str(faiss_dir), embeddings=emb, allow_dangerous_deserialization=True)
        if vectorstore is None:
            vectorstore = past_vs
        else:
            vectorstore.merge_from(past_vs)

        bdir = BACKUP_BASE / category_slug / (cohort if cohort else "all") / ts
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(index_faiss), str(bdir / "index.faiss"))
        shutil.move(str(index_pkl),   str(bdir / "index.pkl"))

    if vectorstore is None:
        print("저장할 벡터스토어가 없습니다. 저장을 건너뜁니다.")
        return

    vectorstore.save_local(str(faiss_dir))
    print(f"저장 완료: {faiss_dir}")

    # 문서 메타 저장(JSONL) — 기존과 합쳐 백업
    docs_dir = DOCS_BASE / category_slug / cohort if cohort else DOCS_BASE / category_slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_jsonl = docs_dir / "doc.jsonl"

    merged_docs: List[LCDocument] = list(docs)
    if doc_jsonl.exists():
        past_docs = load_docs_from_jsonl(str(doc_jsonl))
        merged_docs.extend(past_docs)
        bdir = BACKUP_BASE / category_slug / (cohort if cohort else "all") / ts
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(doc_jsonl), str(bdir / "doc.jsonl"))

    save_docs_to_jsonl(merged_docs, str(doc_jsonl))
    print(f"문서 메타 저장: {doc_jsonl}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="KyungHee-Chatbot: 문서 수집 → 메타 보강 → 인덱싱/병합 유틸리티"
    )
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--category", choices=list(CATEGORIES.keys()), help="단일 카테고리 구축")
    mx.add_argument("--all", action="store_true", help="4개 카테고리를 일괄 구축")

    # 입력 소스 선택
    parser.add_argument("--source", help="입력 디렉터리(예: ./intermediate/regulations_jsonl_upgraded). 지정 안 하면 todo_documents 사용.")
    # 코호트(학부/대학원 시행세칙에서 주로 사용하지만, 규정에도 강제 주입 가능)
    parser.add_argument("--cohort", help="입학년도(예: 2022). 지정 시 Cohort_YYYY로 정규화되어 주입.")

    # 메타 오버라이드
    parser.add_argument("--document-code", dest="document_code", help="documentCode 오버라이드(예: RS)")
    parser.add_argument("--version", dest="version_date", help="versionDate(YYYY-MM-DD) 오버라이드")
    parser.add_argument("--program", help="program 오버라이드(UG/MS/PHD/IME_MS/IME_PHD)")
    parser.add_argument("--track", help="track 오버라이드(GameContents/DataScience/RobotVision/ConvergenceLeader)")
    parser.add_argument("--degree-type", dest="degree_type", help="degreeType 오버라이드(single/multi/minor)")

    # HTTP URI 베이스 네임스페이스
    parser.add_argument("--http-base", dest="http_base", default=DEFAULT_HTTP_URI_BASE,
                        help=f"HTTP 영구 URI 네임스페이스 (기본: {DEFAULT_HTTP_URI_BASE})")

    args = parser.parse_args()

    # 전역 오버라이드 컨텍스트 설정
    CTX.document_code = args.document_code or None
    CTX.version_date  = args.version_date or None
    CTX.program       = args.program or None
    CTX.cohort        = args.cohort or None
    CTX.track         = args.track or None
    CTX.degree_type   = args.degree_type or None
    CTX.http_base     = args.http_base or DEFAULT_HTTP_URI_BASE

    # 대상 카테고리들
    targets = list(CATEGORIES.keys()) if args.all else [args.category]
    source_dir = Path(args.source).resolve() if args.source else None
    if source_dir and not source_dir.exists():
        raise FileNotFoundError(f"--source 경로가 존재하지 않습니다: {source_dir}")

    for slug in targets:
        # cohort는 명시값을 그대로 사용(카테고리 종류와 무관하게 주입 가능)
        apply_cohort = args.cohort

        print("=" * 80)
        label = f"{CATEGORIES[slug]} | {slug}" + (f" | cohort={apply_cohort}" if apply_cohort else "")
        src_label = (str(source_dir) if source_dir else f"todo_documents/{slug}" + (f"/{apply_cohort}" if apply_cohort else ""))
        print(f"[{label}] 인덱스 구축 시작  —  source: {src_label}")

        docs, vs = _process_category(slug, cohort=apply_cohort, source_dir=source_dir)
        _merge_and_save(slug, docs, vs, cohort=apply_cohort)

    print("=" * 80)
    print("모든 작업 완료.")


if __name__ == "__main__":
    main()
