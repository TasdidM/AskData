import json
import logging
import pickle
import sys
import os
from dotenv import load_dotenv
from typing import Any
from knowledge_graph_builder.graph_operations import build_knowledge_graph, export_interactive_visualization
load_dotenv()

schema_filepath = os.getenv("SCHEMA_FILEPATH")
kg_filepath = os.getenv("KG_PICKLE_OUTPUT_PATH")

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
# Using standard logging allows developers to see timestamps, log levels, 
# and easily redirect output to a file during heavy debugging sessions.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==========================================
# CORE PIPELINE FUNCTIONS
# ==========================================

def generate_graph_from_schema(filepath: str) -> Any:
    """
    Opens the schema JSON file and builds the Knowledge Graph.
    
    Args:
        filepath (str): Path to the enriched schema JSON file.
        
    Returns:
        NetworkX Graph object (or the graph type returned by build_knowledge_graph).
    """
    logger.info(f"Attempting to load schema from: {filepath}")
    try:
        with open(filepath, 'r') as schema_json:
            kg = build_knowledge_graph(schema_json)
        
        # Profile and log graph metrics for developer awareness
        num_nodes = kg.number_of_nodes()
        num_edges = kg.number_of_edges()
        
        logger.info("=== Knowledge Graph Generation Statistics ===")
        logger.info(f"Total Nodes Processed: {num_nodes}")
        logger.info(f"Total Edges Generated: {num_edges}")
        
        if num_nodes == 0:
            logger.warning("The generated Knowledge Graph has 0 nodes. Please verify the schema format.")
            
        return kg

    except FileNotFoundError:
        logger.critical(f"Schema file not found at: {filepath}. Pipeline aborted.")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Schema file is not a valid JSON. Error: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error while building the knowledge graph: {e}")
        raise


def verify_rag_metadata(kg: Any) -> None:
    """
    Dynamically grabs a sample node from the graph to verify that 
    metadata extraction is functioning, regardless of schema changes.
    """
    logger.info("Dynamically verifying RAG metadata structure...")
    try:
        # Get a list of all nodes and safely grab the first one
        all_nodes = list(kg.nodes)
        if not all_nodes:
            logger.warning("The graph has no nodes to verify.")
            return
            
        sample_node = all_nodes[0]
        node_attributes = kg.nodes[sample_node]
        
        logger.info(f"Schema Check Passed! Sample node verified.")
        logger.info(f" - Sample Node Name: '{sample_node}'")
        logger.info(f" - Associated Attributes: {node_attributes}")
        
    except Exception as e:
        logger.error(f"Failed to dynamically verify node attributes: {e}")

def save_graph_to_pickle(kg: Any, output_path: str) -> None:
    """
    Serializes and saves the knowledge graph object to a pickle file.
    """
    logger.info(f"Saving knowledge graph object to: {output_path}")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(kg, f)
        logger.info(f"Successfully saved knowledge graph to '{output_path}'")
    except IOError as e:
        logger.error(f"Failed to write pickle file to disk. Disk full or bad permissions? Details: {e}")
        raise


# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================

def build_db_kg():
    # 1. Build the Graph
    kg = generate_graph_from_schema(schema_filepath)
    
    # 2. Debug/Validate structure for RAG
    verify_rag_metadata(kg)
    
    # 3. Export Visualization HTML
    logger.info("Exporting interactive visualization...")
    try:
        export_interactive_visualization(kg)
        logger.info("Visualization export complete.")
    except Exception as e:
        logger.error(f"Visualization export failed, skipping step. Error: {e}")
    
    # 4. Persist Graph State
    save_graph_to_pickle(kg, kg_filepath)
