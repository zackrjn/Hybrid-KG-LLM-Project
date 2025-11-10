import matplotlib
matplotlib.use("Agg")
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

def render_kg(edges: List[Tuple[str, str, str]], out_path: str) -> None:
    G = nx.MultiDiGraph()
    for h, r, t in edges:
        G.add_edge(h, t, label=r)
    # fewer iterations for speed; layout is deterministic with fixed seed
    pos = nx.spring_layout(G, seed=42, iterations=30)
    # fast path: omit labels and edge labels for bulk rendering
    nx.draw(G, pos, with_labels=False, node_size=400)
    plt.savefig(out_path, dpi=100)
    plt.close()
