#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

# TODO: 실제 파이프라인 훅에 맞게 교체
def fake_rag_pipeline(query: str):
    # 실제 앱에서는 chains/get_conversational_rag 호출해 URI/버전 등 추출
    return {"uris": [], "versionDate": None}

def main(p: Path):
    total, passed = 0, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        obj = json.loads(line)
        q = obj["query"]
        expected = obj.get("expected", {})
        got = fake_rag_pipeline(q)
        ok_uri = (not expected.get("uris")) or (set(got.get("uris", [])) & set(expected["uris"]))
        ok_ver = (expected.get("versionDate") in (None, "")) or (got.get("versionDate") == expected["versionDate"])
        if ok_uri and ok_ver:
            passed += 1
        else:
            print(f"[FAIL] {q}\n  expected={expected}\n  got={got}\n")
    rate = 0 if total == 0 else passed / total * 100
    print(f"Passed: {passed}/{total} ({rate:.1f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tests/regression/sample_queries.jsonl")
    args = ap.parse_args()
    main(Path(args.file))
