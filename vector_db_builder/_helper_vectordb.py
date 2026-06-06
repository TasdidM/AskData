import os
import time
import json
import logging
from typing import List, Dict, Any, Optional

# certifi provides Mozilla's root certificates; crucial for secure TLS connections to MongoDB Atlas
import certifi 
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ConnectionFailure, ConfigurationError
from pymongo.operations import SearchIndexModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Configure logging to output time, severity, and the message for easier debugging in production.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file (e.g., MONGO_URI, VECTOR_DB_NAME)
load_dotenv()

# Specify the default embedding model. BAAI/bge-small-en-v1.5 is lightweight, fast,
# making it great for local generation.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


# ==========================================
# 2. DATA PROCESSING LAYER
# ==========================================
def _parse_schema_to_documents(json_filepath: str) -> List[Dict[str, Any]]:
    """
    Parses a JSON schema file and structures it into flat text documents.
    Vector search relies on rich textual context, so we combine table and column 
    metadata into a single string per table.
    """
    logger.info(f"Reading schema file from: {json_filepath}")
    
    try:
        with open(json_filepath, 'r', encoding='utf-8') as file:
            schema_data = json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {json_filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_filepath}: {e}")
        raise

    documents = []
    
    # Assume the JSON structure has a top-level "tables" object
    tables = schema_data.get('tables', {})
    
    for table_name, table_info in tables.items():
        description = table_info.get('description', '')
        synonyms = table_info.get('synonyms', '')
        
        # Construct the context-rich string. This is the exact string the LLM 
        # embedding model will "read" to generate the vector representation.
        content = (
            f"Table Name: {table_name}\n"
            f"Description: {description}\n"
            f"Synonyms: {synonyms}\n"
            f"Columns:\n"
        )
        
        # Append all column definitions to the table's context block
        for col in table_info.get('columns', []):
            content += f" - {col['name']} ({col['type']}): {col['description']} (Synonyms: {col.get('synonyms', '')})\n"
            
        # Store the raw text and metadata. The "content" field is what will be embedded later.
        documents.append({
            "table_name": table_name,
            "description": description,
            "content": content
        })
        
    logger.info(f"Successfully parsed {len(documents)} table documents from schema.")
    return documents


# ==========================================
# 3. DATABASE CONNECTION LAYER
# ==========================================
def connect_mongodb_collection(mongo_uri: str, db_name: str, collection_name: str) -> Optional[Collection]:
    """Establishes and verifies connection to a specific MongoDB collection."""
    if not mongo_uri:
        logger.error("MongoDB URI is empty or missing. Check your .env file.")
        return None

    try:
        logger.info("Attempting connection to MongoDB...")
        
        # Initialize client. 
        # tlsCAFile=certifi.where() prevents SSL handshake errors often seen with Python + Atlas.
        # serverSelectionTimeoutMS=5000 ensures the script doesn't hang indefinitely if offline.
        client = MongoClient(
            mongo_uri, 
            tlsCAFile=certifi.where(), 
            serverSelectionTimeoutMS=5000  
        )
        
        # Force a network round-trip to verify connection validity immediately.
        # MongoClient is lazy by default; 'ping' forces it to actually connect.
        client.admin.command('ping')
        
        db = client[db_name]
        collection = db[collection_name]
        
        logger.info(f"Successfully connected to collection: '{db_name}.{collection_name}'")
        return collection

    except ConnectionFailure as e:
        logger.error(f"Database connection timed out or failed. Details: {e}")
    except ConfigurationError as e:
        logger.error(f"Invalid URI or MongoDB configuration error. Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        
    return None


# ==========================================
# 4. VECTOR DATABASE OPERATIONS
# ==========================================
def populate_mongodb_vector_db(json_filepath: str, collection: Collection) -> None:
    """Generates embeddings and populates the Target MongoDB Collection."""
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    
    # Initialize fastembed model. This downloads the model weights on first run.
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    documents = _parse_schema_to_documents(json_filepath)
    if not documents:
        logger.warning("No documents found to embed. Aborting ingestion loop.")
        return

    logger.info(f"Generating vector embeddings for {len(documents)} entries...")
    for idx, doc in enumerate(documents):
        try:
            # model.embed returns a generator. We use next() to get the first (and only) 
            # item, then convert the numpy array to a native Python list of floats.
            # MongoDB requires vectors to be stored as arrays of standard floats.
            embedding_vector = next(model.embed(doc["content"])).tolist()
            doc["embedding"] = embedding_vector
        except Exception as e:
            logger.error(f"Failed to generate embedding for table '{doc.get('table_name')}': {e}")
            raise e
    
    try:
        # Note: This is a destructive operation. It drops all existing data in the collection
        # to ensure we have a fresh, synchronized state with the current JSON schema.
        logger.info("Clearing existing data from the target collection...")
        delete_result = collection.delete_many({})
        logger.info(f"Dropped {delete_result.deleted_count} stale documents.")
        
        logger.info("Inserting fresh vector documents into Atlas...")
        collection.insert_many(documents)
        logger.info(f"Data ingestion complete. {len(documents)} records written.")
    except Exception as e:
        logger.error(f"Bulk data write execution failed: {e}")
        raise


def create_vector_index(collection: Collection, index_name: str, dimensions: int) -> str:
    """Deploys and monitors a MongoDB Atlas Vector Search Index."""
    
    # Define the structure of the vector search index.
    # 'path' must match the field name where we stored our vectors ("embedding").
    # 'numDimensions' must exactly match the output of our embedding model (e.g., 384).
    # 'similarity' dictates the math used to compare vectors (cosine is standard for semantic search).
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": "cosine"
            }
        ]
    }

    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
        type="vectorSearch"
    )

    try:
        logger.info(f"Inspecting existing search indexes on '{collection.name}'...")
        existing_indexes = list(collection.list_search_indexes())
        
        # Prevent attempting duplicate index deployment if one already exists.
        # MongoDB Atlas throws an error if you try to create an index with identical specs.
        if len(existing_indexes) > 0:
            assigned_name = existing_indexes[0].get('name')
            logger.warning(f"Search index already exists. Skipping allocation. Active Name: {assigned_name}")
            return assigned_name

        logger.info(f"Requesting creation of vector index '{index_name}' from Atlas...")
        result_name = collection.create_search_index(model=search_index_model)
        logger.info(f"Index creation acknowledged by Atlas. Deployment ID/Name: {result_name}")
        
        # Atlas builds vector indexes asynchronously in the background.
        # We use a polling loop to block the script safely until the index is fully built and queryable.
        logger.info("Monitoring Atlas compilation status (takes a few minutes)...")
        while True:
            indices = list(collection.list_search_indexes(name=index_name))
            if indices:
                status = indices[0].get("status")
                logger.info(f"Current Atlas Index Build Status: {status}")
                
                if status == "READY":
                    logger.info(f"SUCCESS! Vector search index '{index_name}' is queryable.")
                    return result_name
                elif status == "FAILED":
                    fail_reason = indices[0].get("queryable_status_desc", "Unknown operational failure")
                    raise RuntimeError(f"Atlas Search Engine build error: {fail_reason}")
            
            # Wait 10 seconds before polling again to avoid spamming the API
            time.sleep(10)
            
    except PyMongoError as pm_err:
        logger.error("Driver level failure encountered during Index coordination.")
        raise pm_err
    except Exception as e:
        logger.error(f"Unexpected operational error during indexing: {e}")
        raise


# ==========================================
# 5. ORCHESTRATION LAYER (EXECUTION ENTRY)
# ==========================================
if __name__ == "__main__":
    # Fetch configurations safely using os.getenv to avoid hardcoding secrets
    MONGO_URI = os.getenv("MONGO_URI", "")
    DB_NAME = os.getenv("VECTOR_DB_NAME")
    COLL_NAME = os.getenv("COLLECTION_NAME")
    SCHEMA_FILE = os.getenv("DB_SCHEMA_JSON_PATH")
    INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
    EMBEDDING_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")

    logger.info("Starting complete MongoDB Vector pipeline debug run...")
    
    # Execute pipeline cleanly step-by-step
    target_collection = connect_mongodb_collection(MONGO_URI, DB_NAME, COLL_NAME)
    
    if target_collection is not None:
        populate_mongodb_vector_db(SCHEMA_FILE, target_collection)
        create_vector_index(target_collection, INDEX_NAME, EMBEDDING_DIMENSIONS)
        logger.info("Pipeline executed flawlessly.")
    else:
        # Halt execution if database connection fails, preventing downstream cascade errors
        logger.critical("Pipeline aborted due to initial database connection failure.")