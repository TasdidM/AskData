"""
Script to initialize and populate a MongoDB vector database.
It loads configuration from environment variables, connects to the database,
processes a JSON schema into text documents for embeddings, and creates a vector index.
"""

import os
import sys
import logging

# Third-party imports
from dotenv import load_dotenv

# Local application imports
from vector_db_builder._helper_vectordb import (
    connect_mongodb_collection, 
    populate_mongodb_vector_db, 
    create_vector_index
)

# Configure basic logging to output progress and errors to the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_vector_database():
    # 1. Load environment variables from the .env file
    logging.info("Loading environment variables...")
    load_dotenv()

    # 2. Retrieve configuration with explicit checks
    # This ensures the script fails immediately and clearly if an env var is missing
    required_env_vars = [
        "DB_SCHEMA_JSON_PATH",
        "MONGO_URI",
        "VECTOR_DB_NAME",
        "COLLECTION_NAME",
        "VECTOR_INDEX_NAME",
        "EMBEDDING_DIMENSIONS"
    ]
    
    config = {}
    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            logging.error(f"Missing required environment variable: {var}")
            sys.exit(1) # Exit the script with an error code
        config[var] = value

    # 3. Connect to the MongoDB Collection
    logging.info(f"Connecting to MongoDB database '{config['VECTOR_DB_NAME']}', collection '{config['COLLECTION_NAME']}'...")
    try:
        mongodb_collection = connect_mongodb_collection(
            mongo_uri=config["MONGO_URI"], 
            db_name=config["VECTOR_DB_NAME"], 
            collection_name=config["COLLECTION_NAME"]
        )
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # 4. Process the schema and populate the vector database
    logging.info(f"Processing JSON schema from {config['DB_SCHEMA_JSON_PATH']} and uploading to MongoDB...")
    try:
        # Transforms detailed schema in JSON to string documents for embeddings
        populate_mongodb_vector_db(config["DB_SCHEMA_JSON_PATH"], mongodb_collection)
    except Exception as e:
        logging.error(f"Failed to populate vector database: {e}")
        sys.exit(1)

    # 5. Create the vector search index
    logging.info(f"Creating vector index '{config['VECTOR_INDEX_NAME']}' with dimension {config['EMBEDDING_DIMENSIONS']}...")
    try:
        create_vector_index(mongodb_collection, config["VECTOR_INDEX_NAME"], config["EMBEDDING_DIMENSIONS"])
        logging.info("Vector index created successfully.")
    except Exception as e:
        logging.error(f"Failed to create vector index: {e}")
        sys.exit(1)

    logging.info("Database setup completed successfully.")
