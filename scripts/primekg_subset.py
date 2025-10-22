#!/usr/bin/env python3
import os
import csv
import json
from typing import Dict, Set


def write_jsonl(items, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--primekg_dir", type=str, default="third_party/PrimeKG")
    parser.add_argument("--out_dir", type=str, default="data/primekg")
    parser.add_argument("--limit_nodes", type=int, default=50000)
    args = parser.parse_args()

    # PrimeKG has CSVs; adapt names if needed
    nodes_csv = os.path.join(args.primekg_dir, "data", "nodes.csv")
    edges_csv = os.path.join(args.primekg_dir, "data", "edges.csv")

    id2text: Dict[str, str] = {}
    allowed_nodes: Set[str] = set()

    if os.path.exists(nodes_csv):
        with open(nodes_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                node_id = row.get("id") or row.get("node_id") or row.get("identifier")
                name = row.get("name") or row.get("label") or node_id
                desc = row.get("description") or ""
                text = f"{name}. {desc}" if desc else str(name)
                if node_id:
                    id2text[node_id] = text
                    if len(allowed_nodes) < args.limit_nodes:
                        allowed_nodes.add(node_id)
    else:
        raise FileNotFoundError(f"Nodes CSV not found at {nodes_csv}")

    triples = []
    if os.path.exists(edges_csv):
        with open(edges_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = row.get("source") or row.get("head")
                t = row.get("target") or row.get("tail")
                r = row.get("relation") or row.get("edge_label") or "related_to"
                if h in allowed_nodes and t in allowed_nodes:
                    triples.append({"head": h, "relation": r, "tail": t})
    else:
        raise FileNotFoundError(f"Edges CSV not found at {edges_csv}")

    write_jsonl(triples, os.path.join(args.out_dir, "triples.jsonl"))
    write_jsonl(({"id": _id, "text": txt} for _id, txt in id2text.items()), os.path.join(args.out_dir, "entity_texts.jsonl"))

    print(f"Wrote subset to {args.out_dir}")


