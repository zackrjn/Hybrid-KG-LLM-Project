import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

def render_kg(edges: List[Tuple[str, str, str]], out_path: str) -> None:
    G = nx.MultiDiGraph()
    for h, r, t in edges:
        G.add_edge(h, t, label=r)
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw(G, pos, with_labels=True, node_size=800, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
