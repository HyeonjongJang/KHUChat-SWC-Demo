# query_parser.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import re

from lark import Lark, Transformer, v_args

GRAMMAR = r"""
?start: query
?query: piece+

?piece: article_range
      | page_range
      | article
      | clause
      | table
      | annex
      | appendix
      | cohort
      | program
      | track
      | degree_type
      | subject_code
      | date
      | WORD      -> kw

# ----- Ranges (explicit separator required) -----
article_range: ART RANGE ART                     -> article_range
page_range   : (P|PAGE) INT (RANGE INT)?         -> page_range

# ----- Singles -----
article : ART                                    -> article
clause  : INT "항" (("및" | "·" | ",") INT "항")*  -> clause
table   : /(표|table)/i                          -> table
annex   : /(부칙)/i                              -> annex
appendix: /(별표|별지)/i                          -> appendix
cohort  : /(20\d{2})\s*학?번?\b|\b\d{2}\s*학?번?\b/i -> cohort
program : /(IME|석사|박사|학부|대학원|MS|PHD|UG)/i -> program
track   : /(게임콘텐츠|데이터사이언스|로봇비전|융합리더|GameContents|DataScience|RobotVision|ConvergenceLeader)/i -> track
degree_type: /(단일전공|복수전공|다전공|부전공)/i -> degree_type
subject_code: /[A-Z]{2,5}\d{3,4}/                  -> subject_code
date    : /(시행일|기준일|effective|since|after|이후|부터)\s*(\d{4}-\d{2}-\d{2})/i -> date

# ----- Tokens -----
RANGE: /(~|–|-)+/
ART  : /제?\s*\d{1,3}\s*조(?:\s*의\s*\d{1,2})?/
P    : /p\./i
PAGE : /페이지/i
WORD : /[^\s]+/

%import common.INT
%import common.WS
%ignore WS
"""

PROG_MAP = {
    "IME": "IME_MS",
    "MS": "MS",
    "석사": "MS",
    "박사": "PHD",
    "PHD": "PHD",
    "학부": "UG",
    "UG": "UG",
    "대학원": "GRAD",
}

@v_args(inline=True)
class QTransform(Transformer):
    def __init__(self):
        self.meta: Dict[str, Any] = {}
        self.hints: Dict[str, Any] = {}
        self.keywords: List[str] = []

    def _parse_art(self, text: str):
        m = re.search(r"\d{1,3}", text)
        if m:
            self.meta["articleNumber"] = int(m.group(0))
        m2 = re.search(r"의\s*(\d{1,2})", text)
        if m2:
            self.meta["clauseNumber"] = int(m2.group(1))

    def article(self, tok):
        self._parse_art(str(tok))
        return None

    def clause(self, *items):
        ints = [int(x) for x in items if hasattr(x, "type") and x.type == "INT"]
        if ints:
            self.meta["clauseNumbers"] = sorted(set(ints))
            self.meta.setdefault("clauseNumber", ints[0])
        return None

    def article_range(self, a1, _sep, a2):
        def _to_num(s: str) -> int:
            m = re.search(r"\d{1,3}", s)
            return int(m.group(0)) if m else None
        left = _to_num(str(a1))
        right = _to_num(str(a2))
        if left is not None and right is not None:
            self.hints.setdefault("articleRanges", []).append((left, right))
            self.meta.setdefault("articleNumber", left)
        return None

    def page_range(self, ptoken, first, maybe_range_int=None):
        a = int(first)
        if maybe_range_int is None:
            self.meta["page"] = a
        else:
            if hasattr(maybe_range_int, "type") and maybe_range_int.type == "INT":
                b = int(maybe_range_int)
            else:
                s = str(maybe_range_int)
                m = re.search(r"\d+", s)
                b = int(m.group(0)) if m else a
            self.hints.setdefault("pageRanges", []).append((a, b))
        return None

    def table(self, *_):
        self.meta["contentType"] = "table"
        self.hints["wants_table"] = True
        return None

    def annex(self, *_):
        self.hints["wants_annex"] = True
        return None

    def appendix(self, *_):
        self.hints["wants_appendix"] = True
        return None

    def cohort(self, tok):
        s = str(tok)
        m4 = re.search(r"(20\d{2})", s)
        m2 = re.search(r"\b(\d{2})\b", s) if not m4 else None
        year = None
        if m4:
            year = m4.group(1)
        elif m2:
            year = f"20{int(m2.group(1)):02d}"
        if year:
            self.meta["cohort"] = f"Cohort_{year}"
        return None

    def program(self, tok):
        s = str(tok).upper()
        for k, v in PROG_MAP.items():
            if k in s or s == k:
                self.meta["program"] = v
                break
        return None

    def track(self, tok):
        s = str(tok)
        ko_map = {
            "게임콘텐츠": "GameContents",
            "데이터사이언스": "DataScience",
            "로봇비전": "RobotVision",
            "융합리더": "ConvergenceLeader",
        }
        for k, v in ko_map.items():
            if k in s:
                self.meta["track"] = v
                return None
        for v in ("GameContents", "DataScience", "RobotVision", "ConvergenceLeader"):
            if v.lower() in s.lower():
                self.meta["track"] = v
                return None
        return None

    def degree_type(self, tok):
        s = str(tok)
        if "단일전공" in s:
            self.meta["degreeType"] = "single"
        elif "다전공" in s or "복수전공" in s:
            self.meta["degreeType"] = "multi"
        elif "부전공" in s:
            self.meta["degreeType"] = "minor"
        return None

    def subject_code(self, tok):
        s = str(tok).upper().strip()
        if re.match(r"^[A-Z]{2,5}\d{3,4}$", s):
            self.meta["subjectCode"] = s
        return None

    def date(self, tok):
        s = str(tok)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            self.hints["refDate"] = m.group(1)
        return None

    def kw(self, tok):
        self.keywords.append(str(tok))
        return None

    def query(self, _):
        if "clauseNumbers" in self.meta and "clauseNumber" not in self.meta:
            self.meta["clauseNumber"] = self.meta["clauseNumbers"][0]
        # reranker 호환: 메타 매칭 키들을 hints 에 동기화 (reranker._meta_score 가 hints 만 봄)
        for k in ("articleNumber", "clauseNumber", "program", "cohort",
                  "track", "degreeType", "subjectCode", "contentType"):
            if k in self.meta:
                self.hints[k] = self.meta[k]
        if self.meta.get("contentType") == "table":
            self.hints["wants_table"] = True
        self.hints["keywords"] = self.keywords
        return {"meta": self.meta, "hints": self.hints}

# Earley parser: robust against ambiguity
_parser = Lark(GRAMMAR, parser="earley", lexer="dynamic_complete")

def parse_query(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        tree = _parser.parse(text or "")
        tx = QTransform()
        res = tx.transform(tree)
        return res["meta"], res["hints"]
    except Exception:
        # Safe fallback: regex router
        from query_router import query_router
        return query_router(text)
