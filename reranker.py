# reranker.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from math import exp
from rank_bm25 import BM25Okapi

# --- rapidfuzz 호환 래퍼 (2.x / 3.x 모두 지원) -------------------------------
try:
    # rapidfuzz 2.x 계열
    from rapidfuzz.string_metric import normalized_levenshtein as _nlev

    def nlev(a: str, b: str) -> float:
        return float(_nlev(str(a or ""), str(b or "")))  # 0.0 ~ 1.0
except Exception:
    # rapidfuzz 3.x+ 계열
    from rapidfuzz.distance import Levenshtein as _Lev

    def nlev(a: str, b: str) -> float:
        # normalized_similarity: 0.0 ~ 1.0
        return float(_Lev.normalized_similarity(str(a or ""), str(b or "")))
# ---------------------------------------------------------------------------


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    x = max(lo, min(hi, x))
    return (x - lo) / (hi - lo)


def _meta_score(md: Dict[str, Any], hints: Dict[str, Any]) -> float:
    sc = 0.0
    # 조 단위 청크가 대부분이라 articleNumber 매칭만으로 강력한 신호
    if hints.get("articleNumber") and md.get("articleNumber") == hints["articleNumber"]:
        sc += 0.80
        if hints.get("clauseNumber") and md.get("clauseNumber") == hints["clauseNumber"]:
            sc += 0.20
    elif hints.get("clauseNumber") and md.get("clauseNumber") == hints["clauseNumber"]:
        sc += 0.40
    if hints.get("program") and md.get("program") == hints["program"]:
        sc += 0.30
    if hints.get("cohort") and md.get("cohort") == hints["cohort"]:
        sc += 0.20
    if hints.get("wants_table") and (md.get("contentType") == "table" or md.get("content_type") == "table"):
        sc += 0.25
    # 신규(소융 데모): 트랙/전공구분/학수번호 일치 가산
    if hints.get("track") and md.get("track") == hints["track"]:
        sc += 0.30
    if hints.get("degreeType") and md.get("degreeType") == hints["degreeType"]:
        sc += 0.20
    if hints.get("subjectCode") and md.get("subjectCode") == hints["subjectCode"]:
        sc += 0.50  # 학수번호 정확 매칭은 강한 신호
    return min(sc, 1.0)


def _version_score(md: Dict[str, Any], ref_date: str | None) -> float:
    v = (md.get("versionDate") or "")[:10]
    if not v:
        return 0.0
    # ref_date가 있으면 ref_date 가까울수록 가산, 없으면 최신(문자열 max) 우대
    if ref_date:
        # 아주 단순한 날짜 근접도(문자열 비교 기반 약식). 필요시 날짜 파싱해서 일수 기반.
        d = 1.0 - (abs(hash(v) - hash(ref_date)) % 1000) / 1000.0
        return max(0.0, d)
    # 최신 우대: 문자열 상 큰 값(YYYY-MM-DD)일수록 큼
    return 0.8


def build_bm25(corpus_texts: List[str]) -> BM25Okapi:
    # 간단 토크나이저: 공백 분할. 한국어 토크나이저 필요시 교체 가능.
    tokenized = [t.split() for t in corpus_texts]
    return BM25Okapi(tokenized)


def bm25_scores(bm25: BM25Okapi, query: str, n: int) -> List[float]:
    return bm25.get_scores(query.split()).tolist()[:n]


def rerank(
    contexts: List[Dict[str, Any]],
    hints: Dict[str, Any],
    query: str,
    weights: Dict[str, float] | None = None
) -> List[Dict[str, Any]]:
    if not contexts:
        return contexts

    # 메타 정확 매칭이 시연 정확도의 핵심.
    # vec·bm25 비중 추가 약화, meta 비중 0.40→0.55 로 강화.
    W = {"vec": 0.25, "bm25": 0.10, "meta": 0.55, "ver": 0.05, "uri": 0.05}
    if weights:
        W.update(weights)

    texts = []
    vecs = []
    mds = []
    for d in contexts:
        md = (d.get("metadata") or {}) if isinstance(d, dict) else getattr(d, "metadata", {}) or {}
        mds.append(md)
        texts.append((d.get("page_content") or d.get("content") or ""))
        vecs.append(d.get("score") or 0.0)  # FAISS 리트리버가 부여한 유사도/거리 역수 등

    # 1) 코사인(또는 리트리버 점수) 정규화
    vmin, vmax = min(vecs), max(vecs)
    if vmin == vmax:
        # 리트리버 점수가 없을 때: 앞쪽(원 리트리버 상위)이 유리하도록 폴백
        n = len(vecs)
        vec_norm = [(n - i) / n for i in range(n)]  # 1.0..(1/n)
    else:
        vec_norm = [_norm01(v, vmin, vmax) for v in vecs]

    # 2) BM25
    bm25 = build_bm25(texts)
    bm = bm25_scores(bm25, query, len(texts))
    bmin, bmax = min(bm), max(bm)
    bm_norm = [_norm01(b, bmin, bmax) for b in bm]

    # 3) 메타/버전/URI
    meta = [_meta_score(md, hints) for md in mds]
    ver = [_version_score(md, hints.get("refDate")) for md in mds]
    uri_hit = []
    target_uri = hints.get("target_uri")
    for md in mds:
        u = md.get("uri") or ""
        uri_hit.append(1.0 if target_uri and u == target_uri else 0.0)

    # 4) 가중합
    scores = []
    for i in range(len(texts)):
        s = (
            W["vec"] * vec_norm[i]
            + W["bm25"] * bm_norm[i]
            + W["meta"] * meta[i]
            + W["ver"] * ver[i]
            + W["uri"] * uri_hit[i]
        )
        scores.append(s)

    # 5) (선택) MMR로 다양성 확보
    k = min(len(contexts), 8)
    lam = 0.65
    selected = []
    cand = list(range(len(contexts)))
    while cand and len(selected) < k:
        if not selected:
            j = max(cand, key=lambda idx: scores[idx])
            selected.append(j)
            cand.remove(j)
            continue

        def sim(a, b):  # 매우 단순한 중복 억제: 내용 nlev
            return nlev(texts[a][:512], texts[b][:512])  # 0..1

        mmr_best, mmr_idx = -1, None
        for idx in cand:
            div = max(sim(idx, s) for s in selected) if selected else 0.0
            val = lam * scores[idx] - (1 - lam) * div
            if val > mmr_best:
                mmr_best, mmr_idx = val, idx
        selected.append(mmr_idx)
        cand.remove(mmr_idx)

    # 재정렬 반영
    ordered = [contexts[i] for i in selected] + [contexts[i] for i in range(len(contexts)) if i not in selected]
    return ordered
