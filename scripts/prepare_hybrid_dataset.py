#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
                     sns_threshold: float = 0.0,
                     sns_model: Optional[str] = None,
                     sns_device: str = "cpu",
                     sns_batch_size: int = 256,
                     emb_cache_npz: Optional[str] = None,
                     render_images: bool = True,
                     max_images: int = 100,
                     num_workers: int = 0,
                     start_idx: int = 0,
                     end_idx: int = -1) -> None:
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    # slice window if requested, then apply limit
    if end_idx is not None and end_idx >= 0:
        triples_window = triples[start_idx:end_idx]
    else:
        triples_window = triples[start_idx:]
    total = min(limit, len(triples_window))
    half = max(1, total // 2) if total >= 2 else 1

    id2text = _load_entity_texts(entity_texts_jsonl)
    adj = _build_adjacency(triples)
    # Optional embedding cache
    emb_cache: Dict[str, object] = {}
    if emb_cache_npz and os.path.exists(emb_cache_npz):
        try:
            import numpy as _np
            loaded = _np.load(emb_cache_npz, allow_pickle=True)
            keys = loaded["keys"].tolist()
            vecs = loaded["vecs"]
            emb_cache = {k: vecs[i] for i, k in enumerate(keys)}
        except Exception:
            emb_cache = {}

    # Initialize ranker (with device/batch size)
    if use_sns:
        model_name = sns_model or "princeton-nlp/sup-simcse-bert-base-uncased"
        ranker = SNSSimilarityRanker(model_name=model_name, device=sns_device, batch_size=sns_batch_size)
    else:
        ranker = None

    with open(out_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        if total == 1:
            h, r, t = triples_window[0]
            img_path = os.path.join(images_dir, f"sample_0.png")
            selected = []
            if use_sns and ranker is not None:
                selected = _select_sns_neighbors(
                    h, r, t, adj, id2text, ranker, top_k=sns_top_k, threshold=sns_threshold
                )
            if not selected:
                selected = [(h, r, t)]
            if render_images and 0 < max_images:
            render_kg(selected, img_path)
            else:
                img_path = None

            lines = [
                f"Question: Given the KG snippet, what is the relation between {h} and {t}?",
                "Relevant KG triples:",
            ]
            for hh, rr, tt in selected:
                lines.append(f"- ({hh}) -[{rr}]-> ({tt})")
            lines.append("Answer:")
            prompt = "\n".join(lines)
            item = {"prompt": prompt, "chosen": r, "rejected": "unknown", "image": img_path}
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")
            return

        tasks: List[Tuple[List[Tuple[str, str, str]], str]] = []
        for idx, (h, r, t) in enumerate(triples_window[:total]):
            img_path = os.path.join(images_dir, f"sample_{idx}.png")
            if use_sns and ranker is not None:
                selected = _select_sns_neighbors(
                    h, r, t, adj, id2text, ranker, top_k=sns_top_k, threshold=sns_threshold
                )
            else:
                selected = [(h, r, t)]
            if render_images and idx < max_images:
                tasks.append((selected, img_path))
            else:
                img_path = None

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

    # After writing JSONL, render images (optionally in parallel)
    if render_images:
        if 'tasks' in locals() and tasks:
            if num_workers and num_workers > 0:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=num_workers) as ex:
                    list(ex.map(lambda x: render_kg(*x), tasks))
            else:
                for edges, path in tasks:
                    render_kg(edges, path)

    # Save embedding cache if requested
    if emb_cache_npz and use_sns:
        try:
            import numpy as _np
            keys = list(emb_cache.keys())
            if keys:
                vecs = _np.stack([emb_cache[k] for k in keys])
                _np.savez_compressed(emb_cache_npz, keys=_np.array(keys, dtype=object), vecs=vecs)
        except Exception:
            pass


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
    parser.add_argument("--sns_model", type=str, default=None, help="SentenceTransformers model (e.g., princeton-nlp/sup-simcse-bert-base-uncased)")
    parser.add_argument("--sns_device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--sns_batch_size", type=int, default=256)
    parser.add_argument("--emb_cache_npz", type=str, default=None, help="Path to NPZ cache for entity embeddings")
    # Rendering and parallelism controls
    parser.add_argument("--no_images", action="store_true", help="Skip rendering KG images")
    parser.add_argument("--max_images", type=int, default=100, help="Render at most N images")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel image workers (0=serial)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) for slicing")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive) for slicing; -1 means till end")
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
        sns_model=args.sns_model,
        sns_device=args.sns_device,
        sns_batch_size=args.sns_batch_size,
        emb_cache_npz=args.emb_cache_npz,
        render_images=(not args.no_images),
        max_images=args.max_images,
        num_workers=args.num_workers,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
