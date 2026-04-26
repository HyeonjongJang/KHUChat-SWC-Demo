#!/usr/bin/env python3
import os, json, argparse, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

HTTP_RE = re.compile(r"^https?://", re.I)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PROGRAMS = {"UG","MS","PHD","IME_MS","IME_PHD"}
CONTENT_TYPES = {"text","table","annex","appendix","table_row_group"}
TRACKS = {"GameContents","DataScience","RobotVision","ConvergenceLeader"}
DEGREE_TYPES = {"single","multi","minor"}
SUBJECT_CODE_RE = re.compile(r"^[A-Z]{2,5}\d{3,4}$")

def _iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for r in data: yield r
        else:
            yield data

def _check_record(meta: Dict[str, Any]) -> List[str]:
    errs = []
    # 1) schema_version
    if not meta.get("schema_version"):
        errs.append("missing schema_version")

    # 2) articleUri/clauseUri HTTP(S)
    for k in ("articleUri","clauseUri"):
        if k in meta and meta[k] is not None and meta[k] != "":
            if not HTTP_RE.match(str(meta[k])):
                errs.append(f"{k} not HTTP(S): {meta[k]}")

    # 3) versionDate + effective keys present
    if not meta.get("versionDate") or not DATE_RE.match(str(meta["versionDate"])):
        errs.append("versionDate invalid/missing")
    for k in ("effectiveFrom","effectiveUntil"):
        if k not in meta: errs.append(f"missing key: {k}")
        else:
            if meta[k] not in (None,"") and not DATE_RE.match(str(meta[k])):
                errs.append(f"{k} invalid date")

    # 4) program/cohort normalized
    prog = meta.get("program")
    if prog not in (None, "") and prog not in PROGRAMS:
        errs.append(f"program not normalized: {prog}")
    coh = meta.get("cohort")
    if coh not in (None, "") and not re.match(r"^Cohort_20\d{2}$", str(coh)):
        errs.append(f"cohort not normalized: {coh}")

    # 5) contentType enum
    ct = meta.get("contentType") or meta.get("content_type")
    if ct not in CONTENT_TYPES:
        errs.append(f"contentType invalid: {ct}")

    # 6) lists
    for k in ("overrides","cites","hasExceptionFor"):
        if k in meta and meta[k] not in (None, []):
            if not isinstance(meta[k], list):
                errs.append(f"{k} must be list")

    # 7) track (신규)
    tr = meta.get("track")
    if tr not in (None, "") and tr not in TRACKS:
        errs.append(f"track invalid: {tr}")

    # 8) degreeType (신규)
    dt = meta.get("degreeType")
    if dt not in (None, "") and dt not in DEGREE_TYPES:
        errs.append(f"degreeType invalid: {dt}")

    # 9) subjectCode (신규)
    scode = meta.get("subjectCode")
    if scode not in (None, "") and not SUBJECT_CODE_RE.match(str(scode)):
        errs.append(f"subjectCode invalid: {scode}")

    # 10) annexNumber (신규)
    an = meta.get("annexNumber")
    if an not in (None, "") and not (isinstance(an, int) and an > 0):
        errs.append(f"annexNumber invalid: {an}")

    return errs

def main(root: Path):
    report = []
    files = list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
    for f in files:
        for rec in _iter_records(f):
            md = rec.get("metadata", rec)
            errs = _check_record(md)
            if errs:
                report.append((f.as_posix(), errs))

    ok = len(report) == 0
    if ok:
        print("✅ All records passed the 6-point schema check.")
        return 0
    else:
        print("❌ Found schema issues:")
        for path, errs in report:
            print(f"- {path}")
            for e in errs:
                print(f"  • {e}")
        return 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs")
    args = ap.parse_args()
    raise SystemExit(main(Path(args.dir)))
