import networkx as nx
from typing import List
from pyvis.network import Network
from itertools import combinations
from networkx.algorithms.approximation import steiner_tree
import json

def build_knowledge_graph(json_file):
    G = nx.Graph()
    data_json = json.load(json_file)
    for table_key, table_data in data_json["tables"].items():

        table_name = table_data["name"]

        # TABLE NODE
        G.add_node(
            table_name,
            node_type="table",
            label=table_name,
        )

        # COLUMN NODES
        for col in table_data["columns"]:

            col_node = f"{table_name}.{col['name']}"

            G.add_node(
                col_node,
                node_type="column",
                label=col['name'],
                data_type=col["type"],
                is_primary_key=(
                    col["name"] in table_data.get("primary_key", [])
                ),
            )

            # Table <-> Column
            G.add_edge(
                table_name,
                col_node,
                relation="has_column"
            )

        # TABLE RELATIONSHIPS
        for rel in table_data.get("relationships", []):

            target_table = rel["to_table"]

            # UNDIRECTED TABLE JOIN EDGE
            G.add_edge(
                table_name,
                target_table,
                relation="JOIN",
                join_from=f"{table_name}.{rel['from_column'][0]}",
                join_to=f"{target_table}.{rel['to_column'][0]}"
            )

    return G

def get_filtered_subgraphs(G: nx.Graph, filtered_tables: list) -> list:
    subgraphs = []

    for u, v in combinations(filtered_tables, 2):
        subgraph = steiner_tree(G, [u, v])
        subgraphs.append(subgraph)

    return subgraphs

def get_subgraphs_joins(subgraphs: List[nx.Graph]) -> str:
    lines = ["The mentioned tables follow the following relations: "]

    for idx, subgraph in enumerate(subgraphs, start=1):

        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges(data=True))

        lines.append(f"\nSubgraph {idx}:")

        # safer endpoint extraction (no assumption of exactly 2 leaves)
        leaves = [n for n, d in subgraph.degree() if d == 1]
        if len(leaves) >= 2:
            lines.append(f"Endpoints: {leaves[0]} <-> {leaves[1]}")

        lines.append(f"Tables used: {nodes}")
        
        for i, j, edge_data in edges:
            join_from = edge_data.get("join_from")
            join_to = edge_data.get("join_to")

            lines.append(
                f"{i} JOIN {j} ON {join_from} = {join_to}"
            )

    return "\n".join(lines)




# 4. YOUR EXACT FUNCTION (Unchanged)
def generate_SQL_rag_prompt(user_query: str, extracted_context: str, sql_service_name: str = "MySQL") -> str:
    """Combines the dynamically retrieved graph context into a strict

    system prompt for the LLM.
    """
    system_prompt = f"""You are a Senior Data Engineer who is professional is SQL writing in {sql_service_name}. 
    Your ONLY task is to translate natural language questions into a valid {sql_service_name} query wrapped inside a JSON object.

    CRITICAL FORMATTING RULES:
    1. Output a valid JSON object with exactly one key: "sql".
    2. The value of "sql" must be a single string containing the raw {sql_service_name} query.
    3. DO NOT include markdown syntax (like ```sql or ```) inside or outside the JSON.
    4. JSON STRING ESCAPING: To prevent JSON parsing errors, always use single quotes (') for string literals inside your SQL query (e.g., WHERE name = 'Alice'). Do not use unescaped double quotes.
    5. No commentary, no explanations. Output ONLY the JSON object.

    Example Output Format:
    {{
        "sql": "SELECT name FROM users WHERE id = 1 AND status = 'active';"
    }}
    """

    user_prompt = f"""The following are the extracted possible relevant tables with their columns name, with subgraph of relations between the tables:
    {extracted_context}

    Instructions:
    Based ONLY on the schema context provided above, generate a valid {sql_service_name} query to answer the user's question. Do not guess or hallucinate table names or columns. Use exact names provided for the tables and columns.

    User Question: {user_query}

    Provide the response as the requested JSON object below:"""

    return system_prompt, user_prompt





def export_interactive_visualization(G, filename="database_schema_kg.html"):
    # Convert standard networkx to interactive PyVis map
    net = Network(
        height="800px", notebook=False, directed=True, heading="Database RAG KG"
    )
    net.from_nx(G)

    # Apply responsive layout mechanics
    net.set_options(
        """
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
    )
    net.save_graph(filename)
    print(f"\n[Success] Graph visualization compiled into '{filename}'")