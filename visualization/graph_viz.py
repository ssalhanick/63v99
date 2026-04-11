# visualization/graph_viz.py

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from pyvis.network import Network

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def get_citation_subgraph(case_id: int, hops: int = 2) -> dict:
    """
    Pull a citation subgraph from Neo4j for a given case ID.
    Returns a dict with 'nodes' and 'edges' lists.
    """
    cypher = f"""
        MATCH path = (c:Case {{id: {case_id}}})-[:CITES*1..{hops}]->(neighbor:Case)
        RETURN path
        LIMIT 100
    """
    nodes = {}  # id -> property dict
    edges = []  # (from_id, to_id) tuples

    with driver.session() as session:
        results = session.run(cypher)
        for record in results:
            path = record["path"]
            for node in path.nodes:
                nid = node["id"]
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "name": node.get("name", f"Case {nid}"),
                        "year": node.get("year", "?"),
                        "court": node.get("court", "?"),
                        "cite_count": node.get("cite_count", 0) or 0,
                        "landmark": node.get("landmark", False),
                        "stub": node.get("stub", False),
                    }
            for rel in path.relationships:
                edges.append((rel.start_node["id"], rel.end_node["id"]))

    return {"nodes": list(nodes.values()), "edges": edges}


def build_pyvis_network(case_id: int, subgraph: dict) -> Network:
    """
    Build and return a PyVis Network object from a subgraph dict.
    Center node (the submitted citation) is highlighted in red.
    Landmark nodes are gold, regular corpus nodes are blue, stubs are gray.
    Node size is proportional to cite_count.
    """
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",       # matches Streamlit dark background
        font_color="white",
    )
    net.barnes_hut()             # physics layout — stable and fast

    node_ids_in_graph = {n["id"] for n in subgraph["nodes"]}

    for node in subgraph["nodes"]:
        nid = node["id"]
        is_center = (nid == case_id)
        is_landmark = node.get("landmark", False)
        is_stub = node.get("stub", False)

        # Color
        if is_center:
            color = "#e74c3c"       # red — submitted citation
        elif is_landmark:
            color = "#f1c40f"       # gold — landmark case
        elif is_stub:
            color = "#7f8c8d"       # gray — stub node
        else:
            color = "#3498db"       # blue — regular corpus case

        # Size: base 15, scale up with cite_count (capped)
        cite_count = min(node["cite_count"], 500)
        size = 15 + (cite_count / 500) * 35

        label = node["name"] if node["name"] else f"Case {nid}"
        # Truncate long names for readability
        if len(label) > 40:
            label = label[:37] + "..."

        tooltip = (
            f"{node['name']}\n"
            f"Year: {node['year']}\n"
            f"Court: {node['court']}\n"
            f"Citations: {node['cite_count']}"
        )

        net.add_node(
            nid,
            label=label,
            title=tooltip,
            color=color,
            size=size,
        )

    for from_id, to_id in subgraph["edges"]:
        # Only add edges where both nodes are present (safety check)
        if from_id in node_ids_in_graph and to_id in node_ids_in_graph:
            net.add_edge(from_id, to_id, color="#555555", arrows="to")

    return net


def render_graph_html(case_id: int, hops: int = 2) -> str:
    """
    Top-level function called by the Streamlit tab.
    Returns the PyVis graph as an HTML string for st.components.v1.html().
    """
    subgraph = get_citation_subgraph(case_id, hops=hops)

    if not subgraph["nodes"]:
        return None  # Caller handles the empty case

    net = build_pyvis_network(case_id, subgraph)

    # Write to a temp file and read back as string
    # (PyVis requires a file path to generate HTML)
    tmp_path = "visualization/_graph_tmp.html"
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    return html, len(subgraph["nodes"]), len(subgraph["edges"])