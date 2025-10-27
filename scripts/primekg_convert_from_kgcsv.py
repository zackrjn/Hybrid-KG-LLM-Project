#!/usr/bin/env python3
import os
import pandas as pd


def convert_kg_to_edges_nodes(kg_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Read as strings to avoid dtype issues on large mixed-type columns
    df = pd.read_csv(kg_csv, dtype=str, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    def pick(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        raise KeyError(f"Missing columns among {keys} in {list(df.columns)}")

    # Support PrimeKG schemas with x_*/y_* columns as well as generic head/tail
    head_col = pick(["head", "source", "subject", "src", "from", "x_id"])  # x_id if present
    tail_col = pick(["tail", "target", "object", "dst", "to", "y_id"])    # y_id if present
    rel_col = pick(["relation", "display_relation", "predicate", "edge_label", "edge", "rel"])

    edges = df[[head_col, rel_col, tail_col]].rename(
        columns={head_col: "head", rel_col: "relation", tail_col: "tail"}
    )
    edges_path = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_path, index=False)

    # Build nodes, preserving names if available (x_name/y_name)
    nodes_parts = []
    # Always include IDs from both ends
    nodes_parts.append(df[[pick(["x_id", head_col.lower()])]].rename(columns={pick(["x_id", head_col.lower()]): "id"}))
    nodes_parts.append(df[[pick(["y_id", tail_col.lower()])]].rename(columns={pick(["y_id", tail_col.lower()]): "id"}))

    x_name_col = cols.get("x_name")
    y_name_col = cols.get("y_name")
    if x_name_col and "x_id" in cols:
        nodes_parts.append(df[[cols["x_id"], x_name_col]].rename(columns={cols["x_id"]: "id", x_name_col: "name"}))
    if y_name_col and "y_id" in cols:
        nodes_parts.append(df[[cols["y_id"], y_name_col]].rename(columns={cols["y_id"]: "id", y_name_col: "name"}))

    nodes = pd.concat(nodes_parts, ignore_index=True).drop_duplicates(subset=["id"])
    if "name" not in nodes.columns:
        nodes["name"] = nodes["id"]
    nodes["name"] = nodes["name"].fillna(nodes["id"])
    nodes["description"] = ""
    nodes = nodes[["id", "name", "description"]].drop_duplicates(subset=["id"])
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


