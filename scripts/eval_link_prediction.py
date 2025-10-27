#!/usr/bin/env python3
import json
import math
from typing import List, Tuple

from src.kg_data import read_triples_jsonl


def mrr_hits_at_k(ranks: List[int], k: int = 10) -> Tuple[float, float]:
    rr = [1.0 / r for r in ranks]
    mrr = sum(rr) / max(1, len(rr))
    hits = sum(1 for r in ranks if r <= k) / max(1, len(ranks))
    return mrr, hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_jsonl", type=str, required=True)
    parser.add_argument("--predictions_jsonl", type=str, required=True, help="JSONL with fields: head, relation, tail_rank")
    args = parser.parse_args()

    gold = read_triples_jsonl(args.triples_jsonl)

    ranks: List[int] = []
    with open(args.predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rnk = int(obj.get("tail_rank", 1000))
            ranks.append(rnk)

    mrr, hits10 = mrr_hits_at_k(ranks, k=10)
    print(json.dumps({"MRR": mrr, "Hits@10": hits10}))


