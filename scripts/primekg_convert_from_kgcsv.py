#!/usr/bin/env python3
import os
import pandas as pd


def convert_kg_to_edges_nodes(kg_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(kg_csv)
    cols = {c.lower(): c for c in df.columns}

    def pick(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        raise KeyError(f"Missing columns among {keys} in {list(df.columns)}")

    head_col = pick(["head", "source", "subject", "src", "from"])
    tail_col = pick(["tail", "target", "object", "dst", "to"])
    rel_col = pick(["relation", "predicate", "edge_label", "edge", "rel"])

    edges = df[[head_col, rel_col, tail_col]].rename(
        columns={head_col: "head", rel_col: "relation", tail_col: "tail"}
    )
    edges_path = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_path, index=False)

    nodes = pd.DataFrame({"id": pd.unique(pd.concat([edges["head"], edges["tail"]]))})
    nodes["name"] = nodes["id"]
    nodes["description"] = ""
    nodes_path = os.path.join(out_dir, "nodes.csv")
    nodes.to_csv(nodes_path, index=False)

    print(f"Wrote {edges_path} and {nodes_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/primekg_raw")
    args = parser.parse_args()

    convert_kg_to_edges_nodes(args.kg_csv, args.out_dir)


