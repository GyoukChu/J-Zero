#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic 2-source RM preference-pair merge.

Concatenate two JSON lists of {"question","chosen","rejected",...}, drop exact
duplicate / degenerate triples, shuffle (seeded), and emit the
{"question","chosen","rejected"} schema rm_train_loop.py consumes. A sidecar
.meta.json records the per-source kept counts (labeled by --name_a/--name_b).
"""
from __future__ import annotations
import argparse, json, os, random
from typing import Dict, List


def _load(path: str) -> List[Dict]:
    if not path or not os.path.exists(path):
        raise SystemExit(f"[merge_pairs] FATAL: input not found: {path}")
    rows = json.load(open(path))
    if not isinstance(rows, list):
        raise SystemExit(f"[merge_pairs] FATAL: {path} is not a JSON list")
    return rows


def _norm(rows: List[Dict], source: str) -> List[Dict]:
    out = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        ch, rj = r.get("chosen", ""), r.get("rejected", "")
        if not (q and isinstance(ch, str) and ch.strip()
                and isinstance(rj, str) and rj.strip() and ch.strip() != rj.strip()):
            continue
        out.append({"question": q, "chosen": ch, "rejected": rj, "_source": source})
    return out


def main(a: argparse.Namespace) -> None:
    rows = _norm(_load(a.pairs_a), a.name_a) + _norm(_load(a.pairs_b), a.name_b)
    if a.pairs_c:
        rows += _norm(_load(a.pairs_c), a.name_c)
    seen, merged = set(), []
    for r in rows:
        key = (r["question"], r["chosen"], r["rejected"])
        if key in seen:
            continue
        seen.add(key); merged.append(r)
    random.Random(a.seed).shuffle(merged)
    from collections import Counter as _C
    bysrc = _C(r["_source"] for r in merged)
    print(f"[merge_pairs] merged {len(merged)} ({dict(bysrc)})", flush=True)
    if a.expected and len(merged) < a.expected:
        msg = f"[merge_pairs] merged {len(merged)} < expected {a.expected}"
        if a.allow_short:
            print("WARN " + msg, flush=True)
        else:
            raise SystemExit("FATAL " + msg + " (pass --allow_short to proceed anyway)")
    out = [{"question": r["question"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in merged]
    os.makedirs(os.path.dirname(a.out_path) or ".", exist_ok=True)
    json.dump(out, open(a.out_path, "w"))
    json.dump({"policy": " + ".join(filter(None, [a.name_a, a.name_b, a.name_c])),
               "n_merged": len(merged), "by_source": dict(bysrc),
               "expected": a.expected, "seed": a.seed},
              open(a.out_path + ".meta.json", "w"), indent=2)
    print(f"[merge_pairs] OK wrote {len(out)} -> {a.out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_a", required=True)
    p.add_argument("--pairs_b", required=True)
    p.add_argument("--pairs_c", default="")
    p.add_argument("--name_a", default="chalneg")
    p.add_argument("--name_b", default="consensus")
    p.add_argument("--name_c", default="synthesis")
    p.add_argument("--out_path", required=True)
    p.add_argument("--expected", type=int, default=0)
    p.add_argument("--allow_short", action="store_true", help="tolerate fewer merged pairs than --expected")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
