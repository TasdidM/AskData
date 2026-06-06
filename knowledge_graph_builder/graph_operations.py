import json
import logging
import networkx as nx
from typing import List, Dict, Any
from itertools import combinations
from pyvis.network import Network
from networkx.algorithms.approximation import steiner_tree

# Configure logging for easier debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_knowledge_graph(json_filepath: str) -> nx.Graph:
    """
    Reads a JSON schema file and constructs a NetworkX graph representing tables,
    columns, and their relationships (JOINs).
    
    Args:
        json_filepath (str): The file path to the JSON schema file.
        
    Returns:
        nx.Graph: A constructed Knowledge Graph of the database schema.
    """
    G = nx.Graph()
    
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load JSON file at {json_filepath}: {e}")
        return G

    logger.info("Building Knowledge Graph from JSON data...")
    
    for table_key, table_data in data_json.get("tables", {}).items():
        table_name = table_data.get("name", table_key)

        # 1. Add Table Node
        G.add_node(
            table_name,
            node_type="table",
            label=table_name,
        )

        # 2. Add Column Nodes and Edges to Table
        for col in table_data.get("columns", []):
            col_name = col.get("name")
            col_node = f"{table_name}.{col_name}"

            G.add_node(
                col_node,
                node_type="column",
                label=col_name,
                data_type=col.get("type"),
                is_primary_key=(col_name in table_data.get("primary_key", [])),
            )

            # Create an edge between the table and its column
            G.add_edge(
                table_name,
                col_node,
                relation="has_column"
            )

        # 3. Add Table Relationships (JOINs)
        for rel in table_data.get("relationships", []):
            target_table = rel.get("to_table")
            
            # Avoid errors if schema format is slightly off
            if not target_table or not rel.get("from_column") or not rel.get("to_column"):
                logger.warning(f"Skipping malformed relationship in table {table_name}")
                continue

            join_from_col = f"{table_name}.{rel['from_column'][0]}"
            join_to_col = f"{target_table}.{rel['to_column'][0]}"

            # Undirected Table JOIN edge
            G.add_edge(
                table_name,
                target_table,
                relation="JOIN",
                join_from=join_from_col,
                join_to=join_to_col
            )

    logger.info(f"Successfully built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def get_filtered_subgraphs(G: nx.Graph, filtered_tables: List[str]) -> List[nx.Graph]:
    """
    Finds paths (Steiner Trees) between all combinations of requested tables.
    
    Args:
        G (nx.Graph): The complete Knowledge Graph.
        filtered_tables (List[str]): A list of table names to connect.
        
    Returns:
        List[nx.Graph]: A list of subgraphs representing the connections.
    """
    subgraphs = []
    
    # Verify tables actually exist in graph to prevent silent failures
    valid_tables = [t for t in filtered_tables if G.has_node(t)]
    if len(valid_tables) != len(filtered_tables):
        logger.warning(f"Some tables were not found in the graph: {set(filtered_tables) - set(valid_tables)}")

    for u, v in combinations(valid_tables, 2):
        try:
            subgraph = steiner_tree(G, [u, v])
            subgraphs.append(subgraph)
        except Exception as e:
            logger.error(f"Failed to find Steiner tree between {u} and {v}: {e}")

    return subgraphs

def get_subgraphs_joins(subgraphs: List[nx.Graph]) -> str:
    """
    Parses subgraphs to extract and format SQL JOIN relationships as a readable string.
    
    Args:
        subgraphs (List[nx.Graph]): Subgraphs representing table connections.
        
    Returns:
        str: A formatted string detailing the JOIN paths.
    """
    lines = ["The mentioned tables follow the following relations: "]

    for idx, subgraph in enumerate(subgraphs, start=1):
        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges(data=True))

        lines.append(f"\nSubgraph {idx}:")

        # Safer endpoint extraction: find nodes with only 1 degree (leaves)
        leaves = [n for n, d in subgraph.degree() if d == 1]
        if len(leaves) >= 2:
            lines.append(f"Endpoints: {leaves[0]} <-> {leaves[1]}")

        lines.append(f"Tables used: {nodes}")
        
        # Extract the JOIN conditions from edge attributes
        for i, j, edge_data in edges:
            if edge_data.get("relation") == "JOIN":
                join_from = edge_data.get("join_from", "UNKNOWN")
                join_to = edge_data.get("join_to", "UNKNOWN")
                lines.append(f"{i} JOIN {j} ON {join_from} = {join_to}")

    return "\n".join(lines)

def export_interactive_visualization(G: nx.Graph, filename: str = "assets/database_schema_kg.html") -> None:
    """
    Converts a NetworkX graph into an interactive HTML visualization using PyVis.
    
    Args:
        G (nx.Graph): The networkx graph to visualize.
        filename (str): The destination HTML file path.
    """
    try:
        net = Network(height="800px", notebook=False, directed=True, heading="Database RAG KG")
        net.from_nx(G)

        # Apply custom physics settings for better layout stability
        physics_options = """
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -15000,
              "centralGravity": 0.2,
              "springLength": 120,
              "springConstant": 0.05
            },
            "stabilization": {"iterations": 150}
          }
        }
        """
        net.set_options(physics_options)
        net.save_graph(filename)
        logger.info(f"[Success] Graph visualization compiled into '{filename}'")
        
    except Exception as e:
        logger.error(f"Failed to export visualization: {e}")