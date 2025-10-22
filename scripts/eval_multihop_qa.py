#!/usr/bin/env python3
import json
from typing import List


def accuracy(preds: List[str], golds: List[str]) -> float:
    correct = 0
    for p, g in zip(preds, golds):
        correct += 1 if p.strip().lower() == g.strip().lower() else 0
    return correct / max(1, len(golds))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_jsonl", type=str, required=True, help="JSONL with fields: prompt, answer")
    parser.add_argument("--pred_jsonl", type=str, required=True, help="JSONL with field: prediction")
    args = parser.parse_args()

    golds: List[str] = []
    preds: List[str] = []

    with open(args.gold_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            golds.append(str(obj.get("answer", "")))

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds.append(str(obj.get("prediction", "")))

    acc = accuracy(preds, golds)
    print(json.dumps({"Accuracy": acc}))


