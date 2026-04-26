#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Iterable, Optional, List

PROGRAM_MAP = {
    "관광대학원": "GraduateSchoolOfTourism",
    "GraduateSchoolOfTourism": "GraduateSchoolOfTourism",
    "GST": "GraduateSchoolOfTourism",
}
CATEGORY_SET = {"regulations", "undergrad_rules", "grad_rules", "academic_system"}

INT_RE  = re.compile(r"\d+")
YEAR_RE = re.compile(r"(20\d{2})")

def _korean_article_to_int(s: Optional[str]) -> Optional[int]:
    if not s: return None
    m = INT_RE.search(s); return int(m.group(0)) if m else None

def _infer_article(meta: Dict) -> Optional[int]:
    for k in ("articleNumber", "article_number"):
        v = meta.get(k)
        if isinstance(v, int): return v
        if isinstance(v, str):
            m = INT_RE.search(v)
            if m: return int(m.group(0))
    return _korean_article_to_int(meta.get("article_number"))

def _infer_clause(meta: Dict) -> Optional[int]:
    v = meta.get("articleSub")
    if v in (None, "", "null"): return None
    try: return int(v)
    except: return None

def _infer_label(meta: Dict, article: Optional[int]) -> Optional[str]:
    title = meta.get("article_title") or meta.get("articleTitle")
    if not article and not title: return None
    if article and title: return f"제{article}조 {title}".strip()
    if article: return f"제{article}조"
    return title

def _pick_page(meta: Dict) -> Optional[int]:
    for k in ("page", "page_number"):
        v = meta.get(k)
        if isinstance(v, int): return v
        if isinstance(v, str) and v.isdigit(): return int(v)
    return None

def _guess_category_from_path(p: Path) -> Optional[str]:
    parts = [*p.parts]
    for cat in CATEGORY_SET:
        if cat in parts: return cat
    return None

def _guess_cohort_from_path(p: Path) -> Optional[str]:
    for part in p.parts:
        m = YEAR_RE.search(part)
        if m: return m.group(1)
    return None

def _guess_program_from_path(p: Path) -> Optional[str]:
    for part in p.parts:
        if part in PROGRAM_MAP: return PROGRAM_MAP[part]
    return None

def _read_text_with_fallback(path: Path) -> Optional[str]:
    encs = ["utf-8", "utf-16", "cp949"]
    for enc in encs:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    # 최후의 수단: 바이너리 읽고 디코딩 오류 무시
    try:
        return path.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return None

def _json_objects_from_text(text: str) -> List[Dict]:
    text = text.strip()
    if not text: return []
    # 1) 단일 JSON 객체/배열 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):  return [obj]
        if isinstance(obj, list):  return obj
    except Exception:
        pass
    # 2) JSONL/NDJSON 시도
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

def _load_items_from_file(path: Path) -> List[Dict]:
    txt = _read_text_with_fallback(path)
    if txt is None: return []
    items = _json_objects_from_text(txt)
    # 3) 여전히 못 읽었으면, 첫 '{'~마지막 '}' 구간만 잘라 재시도(간헐적 BOM/머릿글 제거용)
    if not items and "{" in txt and "}" in txt:
        try:
            cut = txt[txt.find("{"): txt.rfind("}")+1]
            items = _json_objects_from_text(cut)
        except Exception:
            pass
    return items

def _make_uri_http(code: str, vdate: str, article: int, clause: Optional[int]) -> str:
    base = f"https://kg.khu.ac.kr/reg/{code}/{vdate}/art{article}"
    return f"{base}/cl{clause}" if clause else base

def _make_urn(code: str, vdate: str, article: int, clause: Optional[int]) -> str:
    urn = f"urn:khu:reg:{code}:{vdate}:art{article}"
    return f"{urn}:cl{clause}" if clause else urn

def _make_id_http(md5: str, article: Optional[int], clause: Optional[int]) -> str:
    if article:
        base = f"https://kg.khu.ac.kr/id/{md5}/art{article}"
        return f"{base}/cl{clause}" if clause else base
    return f"https://kg.khu.ac.kr/id/{md5}"

def convert_file(
    fpath: Path,
    default_category: Optional[str],
    default_program: Optional[str],
    default_cohort: Optional[str],
    default_code: Optional[str],
    default_effective_from: Optional[str],
):
    items = _load_items_from_file(fpath)
    print(f"[SCAN] {fpath} -> {len(items)} item(s)")
    cat_from_path    = _guess_category_from_path(fpath)
    cohort_from_path = _guess_cohort_from_path(fpath)
    prog_from_path   = _guess_program_from_path(fpath)

    for it in items:
        meta = it.get("metadata", {})
        md5 = meta.get("md5") or "nohash"
        article = _infer_article(meta)
        clause  = _infer_clause(meta)
        page    = _pick_page(meta)
        label   = _infer_label(meta, article)
        source  = meta.get("sourceFile") or meta.get("document_title")

        category = default_category or cat_from_path or None
        program  = default_program  or prog_from_path or None
        cohort   = default_cohort   or cohort_from_path or None

        m: Dict = {
            "category": category,
            "program":  program,
            "cohort":   cohort,
            "article":  int(article) if article else None,
            "clause":   int(clause)  if clause  is not None else None,
            "label":    label,
            "source":   source,
            "page":     page,
            "md5":      md5,
            "cites": [],
            "overrides": [],
            "hasExceptionFor": [],
        }

        code  = default_code
        vdate = default_effective_from
        if code and vdate and article:
            http_uri = _make_uri_http(code, vdate, article, clause)
            urn_uri  = _make_urn(code,  vdate, article, clause)
            if clause:
                m["clauseUri"]  = http_uri
                m["articleUri"] = _make_uri_http(code, vdate, article, None)
            else:
                m["articleUri"] = http_uri
            m["uri"] = urn_uri
            m["effectiveFrom"] = vdate
        else:
            m["uri"] = _make_id_http(md5, article, clause)

        yield m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="../past_documents",
                    help="원시 JSON 루트 (기본: ../past_documents)")
    ap.add_argument("--out", default="meta.jsonl",
                    help="출력 메타 JSONL (기본: meta.jsonl)")
    ap.add_argument("--code", default=None,
                    help="정식 규정코드(예: AA, GST 등)")
    ap.add_argument("--effective-from", default=None,
                    help="시행일 YYYY-MM-DD")
    ap.add_argument("--program", default=None,
                    help="기본 program")
    ap.add_argument("--category", default=None, choices=list(CATEGORY_SET),
                    help="기본 category")
    ap.add_argument("--cohort", default=None,
                    help="기본 cohort")
    ap.add_argument("--glob", default="**/*.json,**/*.JSON,**/*.Json,**/*.jsonl,**/*.ndjson",
                    help="콤마로 구분된 glob 패턴")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    patterns = [g.strip() for g in args.glob.split(",") if g.strip()]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(in_root.glob(pat)))

    print(f"[INFO] in_root={in_root}")
    print(f"[INFO] patterns={patterns}")
    print(f"[INFO] matched_files={len(files)}")
    for fp in files[:10]:
        print(f"  - {fp}")
    if len(files) > 10:
        print(f"  ... (+{len(files)-10} more)")

    if not files:
        print(f"[WARN] No files matched under {in_root} with patterns: {patterns}")
        return 0

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with outp.open("w", encoding="utf-8") as fw:
        for fp in files:
            for meta in convert_file(
                fp,
                default_category=args.category,
                default_program=args.program,
                default_cohort=args.cohort,
                default_code=args.code,
                default_effective_from=args.effective_from,
            ):
                fw.write(json.dumps(meta, ensure_ascii=False) + "\n")
                total += 1

    print(f"[OK] wrote {outp} ({total} items)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
