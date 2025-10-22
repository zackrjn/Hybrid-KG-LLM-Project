#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Tuple, Optional

from src.kg_data import read_triples_jsonl
from src.kg_visualize import render_kg
from src.sns_ranker import SNSSimilarityRanker


def _load_entity_texts(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    id2text: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _id = str(obj.get("id", ""))
            _tx = str(obj.get("text", _id))
            if _id:
                id2text[_id] = _tx
    return id2text


def _build_adjacency(triples: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for h, r, t in triples:
        adj.setdefault(h, []).append((r, t))
        adj.setdefault(t, [])  # ensure key exists
    return adj


def _select_sns_neighbors(h: str,
                          r: str,
                          t: str,
                          adj: Dict[str, List[Tuple[str, str]]],
                          id2text: Dict[str, str],
                          ranker: SNSSimilarityRanker,
                          top_k: int,
                          threshold: float) -> List[Tuple[str, str, str]]:
    # gather one-hop candidates around h and t
    cands: List[Tuple[str, str]] = []
    for rel, nb in adj.get(h, []):
        cands.append((rel, nb))
    for rel, nb in adj.get(t, []):
        cands.append((rel, nb))
    # de-duplicate by neighbor id, keep first relation seen
    seen: Dict[str, str] = {}
    for rel, nb in cands:
        if nb not in seen:
            seen[nb] = rel
    if not seen:
        return []

    h_text = id2text.get(h, h)
    t_text = id2text.get(t, t)
    q_text = f"{h_text} {r} {t_text}"
    cand_ids = list(seen.keys())
    cand_texts = [id2text.get(cid, cid) for cid in cand_ids]
    ranked = ranker.rank_neighbors(q_text, cand_texts, top_k=top_k, similarity_threshold=threshold)
    selected: List[Tuple[str, str, str]] = []
    for idx, _sim in ranked:
        nb_id = cand_ids[idx]
        rel = seen[nb_id]
        # prefer triples anchored at h or t
        if nb_id == t:
            selected.append((h, r, t))
        else:
            # attach to whichever side provided this neighbor
            if any(nb_id == nb for _, nb in adj.get(h, [])):
                selected.append((h, rel, nb_id))
            elif any(nb_id == nb for _, nb in adj.get(t, [])):
                selected.append((t, rel, nb_id))
    # ensure the anchor triple is present
    if (h, r, t) not in selected:
        selected.insert(0, (h, r, t))
    return selected


def build_demo_pairs(triples: List[Tuple[str, str, str]],
                     out_dir: str,
                     limit: int = 50,
                     use_sns: bool = False,
                     entity_texts_jsonl: Optional[str] = None,
                     sns_top_k: int = 5,
                     sns_threshold: float = 0.0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    half = max(1, limit // 2)

    id2text = _load_entity_texts(entity_texts_jsonl)
    adj = _build_adjacency(triples)
    ranker = SNSSimilarityRanker() if use_sns else None

    with open(out_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        for idx, (h, r, t) in enumerate(triples[:limit]):
            img_path = os.path.join(images_dir, f"sample_{idx}.png")
            if use_sns and ranker is not None:
                selected = _select_sns_neighbors(
                    h, r, t, adj, id2text, ranker, top_k=sns_top_k, threshold=sns_threshold
                )
            else:
                selected = [(h, r, t)]
            render_kg(selected, img_path)

            lines = [
                f"Question: Given the KG snippet, what is the relation between {h} and {t}?",
                "Relevant KG triples:",
            ]
            for hh, rr, tt in selected:
                lines.append(f"- ({hh}) -[{rr}]-> ({tt})")
            lines.append("Answer:")
            prompt = "\n".join(lines)
            chosen = r
            rejected = "unknown"

            item = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "image": img_path,
            }
            if idx < half:
                f_train.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                f_val.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/hybrid")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--use_sns", action="store_true")
    parser.add_argument("--entity_texts_jsonl", type=str, default=None)
    parser.add_argument("--sns_top_k", type=int, default=5)
    parser.add_argument("--sns_threshold", type=float, default=0.0)
    args = parser.parse_args()

    triples = read_triples_jsonl(args.triples_jsonl)
    build_demo_pairs(
        triples,
        args.out_dir,
        limit=args.limit,
        use_sns=args.use_sns,
        entity_texts_jsonl=args.entity_texts_jsonl,
        sns_top_k=args.sns_top_k,
        sns_threshold=args.sns_threshold,
    )
