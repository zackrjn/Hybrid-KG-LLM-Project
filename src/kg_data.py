from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import json
import random


Triple = Tuple[str, str, str]


def read_triples_jsonl(path: str,
                       head_field: str = "head",
                       relation_field: str = "relation",
                       tail_field: str = "tail") -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            h = str(obj[head_field])
            r = str(obj[relation_field])
            t = str(obj[tail_field])
            triples.append((h, r, t))
    return triples


def read_triples_tsv(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
    return triples


def split_triples(triples: List[Triple],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  seed: int = 42) -> Dict[str, List[Triple]]:
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    rng = random.Random(seed)
    shuffled = triples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def iter_batches(items: List[Triple], batch_size: int) -> Iterator[List[Triple]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


