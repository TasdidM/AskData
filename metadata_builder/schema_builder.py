import logging
from typing import Dict, Any, Optional

# Assuming these are imported from your other modules
from metadata_builder.LLM_pydantic import (
    ColumnMetadata,
    RelationshipMetadata,
    TableMetadata,
    DatabaseSchema
)
from functions.llm_service import generate_verbose_descriptions

# Configure logging to make debugging much easier for the team.
# You can change INFO to DEBUG when you need more granularity.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _enrich_single_table(table_name: str, table_info: Dict[str, Any]) -> Optional[TableMetadata]:
    """
    Helper function to process a single table. 
    Isolating this logic makes unit testing and debugging individual tables much easier.
    
    Args:
        table_name (str): The name of the table in the database.
        table_info (Dict[str, Any]): The raw metadata for this specific table.
        
    Returns:
        Optional[TableMetadata]: The enriched Pydantic model for the table, or None if the LLM fails.
    """
    logger.info(f"Generating LLM description for table: '{table_name}'")
    
    try:
        # 1. Fetch AI descriptions. Wrapped in a try-except because LLM calls can fail or timeout.
        ai_desc = generate_verbose_descriptions(table_name, table_info)
    except Exception as e:
        logger.error(f"Failed to generate LLM descriptions for table '{table_name}'. Error: {e}")
        return None

    # 2. Build fast lookup dictionaries to map AI-generated data back to the raw columns
    # Using dictionary comprehensions here ensures O(1) lookup times below.
    desc_lookup = {col.name: col.description for col in ai_desc.columns}
    synonyms_lookup = {col.name: col.synonyms for col in ai_desc.columns}

    processed_columns = []
    
    # 3. Iterate through the raw columns and merge in the AI data
    for raw_col in table_info.get('columns', []):
        col_name = raw_col.get('name')
        
        if not col_name:
            logger.warning(f"Found a column without a name in table '{table_name}'. Skipping.")
            continue

        # Safely fetch AI data with clear fallbacks in case the LLM missed a column
        col_description = desc_lookup.get(col_name, "No description generated.")
        col_synonyms = synonyms_lookup.get(col_name, []) # Assuming synonyms expects a list

        processed_columns.append(ColumnMetadata(
            name=col_name,
            type=raw_col.get('type', 'UNKNOWN'),
            nullable=raw_col.get('nullable', True),
            description=col_description,
            synonyms=col_synonyms
        ))

    # 4. Map relationships (fallback to an empty list if none exist)
    processed_relationships = [
        RelationshipMetadata(**rel) for rel in table_info.get('relationships', [])
    ]

    # 5. Construct and return the final Pydantic model for this table
    return TableMetadata(
        name=table_name,
        description=ai_desc.description,
        synonyms=ai_desc.synonyms,
        columns=processed_columns,
        primary_key=table_info.get('primary_key', []),
        relationships=processed_relationships
    )


def build_enriched_schema(db_metadata: Dict[str, Any]) -> DatabaseSchema:
    """
    Iterates through raw database metadata, enriches it with LLM-generated 
    table and column descriptions, and constructs a validated Pydantic schema.
    
    Args:
        db_metadata (Dict[str, Any]): The raw extracted database schema information.
        
    Returns:
        DatabaseSchema: A fully populated Pydantic model containing the enriched schema.
    """
    final_database_schema = {}
    
    logger.info(f"Starting schema enrichment for {len(db_metadata)} tables.")

    for table_name, table_info in db_metadata.items():
        enriched_table = _enrich_single_table(table_name, table_info)
        
        if enriched_table:
            final_database_schema[table_name] = enriched_table
        else:
            logger.warning(f"Table '{table_name}' was skipped due to processing errors.")

    logger.info("Schema enrichment complete.")
    
    # Wrap and return the final container
    return DatabaseSchema(tables=final_database_schema)


def export_enriched_schema_to_json(schema: DatabaseSchema, filepath: str = "assets/enriched_schema.json") -> str:
    """
    Helper function to safely export the Pydantic DatabaseSchema object to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # model_dump_json is the Pydantic v2 standard (use .json() if you are on v1)
            f.write(schema.model_dump_json(indent=2))
        logger.info(f"Successfully exported enriched schema to {filepath}")
    except IOError as e:
        logger.error(f"Failed to write schema to {filepath}. Error: {e}")
        
    return filepath