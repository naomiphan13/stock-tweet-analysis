import pandas as pd
import networkx as nx
from pyvis.network import Network
from IPython.display import IFrame
import os

CURRENT_DIR = os.getcwd()
DATA_DIR = f"{CURRENT_DIR}/interactive_neo4j/all"

def visualize_neo4j_csv(
    node_csv: str,
    rel_csv: str,
    *,
    node_id_col: str = "~id",
    node_label_col: str | None = None,
    node_label_fallbacks: list[str] | None = None,
    node_type_col: str = "~labels",
    rel_start_col: str = "~start_id",
    rel_end_col: str = "~end_id",
    rel_type_col: str = "~type",
    max_nodes: int | None = None,
    html_output: str = "neo4j_graph.html",
    directed: bool = True,
):
    """
    Load Neo4j-exported node & relationship CSVs and show an interactive graph (PyVis).

    Parameters
    ----------
    node_csv : str
        Path to the nodes CSV file.
    rel_csv : str
        Path to the relationships CSV file.
    node_id_col : str
        Column name in node_csv that uniquely identifies a node (usually '~id').
    node_label_col : str | None
        Column to use as the main visible label (e.g., 'name', 'symbol').
        If None, try node_label_fallbacks then node_type_col.
    node_label_fallbacks : list[str] | None
        List of node columns to try in order if node_label_col is None or missing.
    node_type_col : str
        Column that stores Neo4j labels (usually '~labels').
    rel_start_col : str
        Column in rel_csv with the source node ID (usually '~start_id').
    rel_end_col : str
        Column in rel_csv with the target node ID (usually '~end_id').
    rel_type_col : str
        Column that stores relationship type (usually '~type').
    max_nodes : int | None
        If set, only keep the first max_nodes nodes (and edges between them).
    html_output : str
        Output HTML filename for PyVis.
    directed : bool
        If True, uses a directed graph; otherwise undirected.
    """
    # Sensible defaults for label columns if not provided
    if node_label_fallbacks is None:
        node_label_fallbacks = ["name", "symbol", "username", "id"]

    # Load CSVs
    nodes = pd.read_csv(node_csv)
    rels = pd.read_csv(rel_csv)

    # Optionally limit number of nodes
    if max_nodes is not None and len(nodes) > max_nodes:
        nodes = nodes.head(max_nodes)
        allowed_ids = set(nodes[node_id_col])
        rels = rels[rels[rel_start_col].isin(allowed_ids) &
                    rels[rel_end_col].isin(allowed_ids)]

    # Build NetworkX graph
    G = nx.DiGraph() if directed else nx.Graph()

    # ---- Add nodes ----
    for _, row in nodes.iterrows():
        node_key = row[node_id_col]

        # Decide label:
        label = None
        # 1) explicit node_label_col
        if node_label_col and node_label_col in row:
            val = row[node_label_col]
            if pd.notna(val):
                label = str(val)
        # 2) fallbacks
        if label is None:
            for col in node_label_fallbacks:
                if col in row and pd.notna(row[col]):
                    label = str(row[col])
                    break
        # 3) fallback to node type or ID
        if label is None:
            if node_type_col in row and pd.notna(row[node_type_col]):
                label = str(row[node_type_col])
            else:
                label = str(node_key)

        attrs = row.to_dict()
        attrs["__label"] = label  # store chosen label explicitly
        G.add_node(node_key, **attrs)

    # ---- Add edges ----
    for _, row in rels.iterrows():
        src = row[rel_start_col]
        tgt = row[rel_end_col]

        # Skip edges involving nodes we don't have (after filtering)
        if src not in G.nodes or tgt not in G.nodes:
            continue

        rel_type = None
        if rel_type_col in row and pd.notna(row[rel_type_col]):
            rel_type = str(row[rel_type_col])

        attrs = row.to_dict()
        attrs["__rel_type"] = rel_type
        G.add_edge(src, tgt, **attrs)

    # ---- Visualize with PyVis ----
    net = Network(
        notebook=True,
        directed=directed,
        height="600px",
        width="100%",
    )

    # Add nodes to PyVis
    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("__label", str(node_id))
        title = "<br>".join(
            f"{k}: {v}" for k, v in attrs.items() if not k.startswith("__")
        )
        net.add_node(node_id, label=label, title=title)

    # Add edges to PyVis
    for u, v, attrs in G.edges(data=True):
        title_parts = []
        if attrs.get("__rel_type"):
            title_parts.append(f"type: {attrs['__rel_type']}")
        # include other properties but skip internals
        for k, val in attrs.items():
            if k.startswith("__"):
                continue
            title_parts.append(f"{k}: {val}")
        title = "<br>".join(title_parts)
        net.add_edge(u, v, title=title)

    net.show(html_output)
    return IFrame(html_output, width="100%", height="600px")

visualize_neo4j_csv(
    
    node_csv=f"{DATA_DIR}/node-export.csv",
    rel_csv=f"{DATA_DIR}/relationship-export.csv",
    # choose a nice main label if you want; otherwise it will auto-pick
    node_label_col=None,           # use auto fallback (name, symbol, username, id, ...)
    html_output="neo4j_graph.html",
    max_nodes=300,                 # optional – keep graphs manageable
)
