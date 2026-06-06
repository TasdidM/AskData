import os
import json
from metadata_builder.extract_metadata import extract_db_metadata, generate_schema_from_db, export_schema_to_json
from metadata_builder.db_connection import get_database_url
from metadata_builder.schema_builder import build_enriched_schema, export_enriched_schema_to_json

# Define the path for the cached schema dict
ASSETS_DIR = "assets"
CACHED_SCHEMA_PATH = os.path.join(ASSETS_DIR, "db_base_schema.json")

def run_enriched_extraction():
    """Runs the enriched schema metadata extraction."""
    metadata = extract_db_metadata(get_database_url())
    schema = build_enriched_schema(metadata)
    db_enriched_schema_path = export_enriched_schema_to_json(schema)
    print(f"Database enriched schema saved to filepath: {db_enriched_schema_path}")

def run_basic_extraction_and_save(schema_dict):
    """Runs the basic extraction and saves the state to the assets folder."""
    db_base_schema_path = export_schema_to_json(schema_dict)
    
    # Save the dictionary to the assets folder for future comparison
    with open(CACHED_SCHEMA_PATH, "w") as f:
        json.dump(schema_dict, f, indent=4)
        
    print(f"Database base schema saved to filepath: {db_base_schema_path}")

def build_db_metadata():
    # Ensure the assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Always generate the current schema dictionary to check for changes
    current_schema_dict = generate_schema_from_db(get_database_url())

    # 1. Check if the saved schema dictionary exists in the assets folder
    if not os.path.exists(CACHED_SCHEMA_PATH):
        print("No saved schema found in assets folder. Running initial extractions...")
        run_enriched_extraction()
        run_basic_extraction_and_save(current_schema_dict)

    # 2. If it does exist, compare current vs. saved
    else:
        with open(CACHED_SCHEMA_PATH, "r") as f:
            try:
                saved_schema_dict = json.load(f)
            except json.JSONDecodeError:
                saved_schema_dict = {}

        if current_schema_dict != saved_schema_dict:
            print("Schema changes detected. Updating metadata extractions...")
            run_enriched_extraction()
            run_basic_extraction_and_save(current_schema_dict)
        else:
            print("No schema changes detected. Extraction skipped.")

