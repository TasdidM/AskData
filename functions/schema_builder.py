# schema_builder.py
import json
from typing import Dict, Any

# Assuming these are imported from your other modules
from functions.llm_pydantic import (
    ColumnMetadata,
    RelationshipMetadata,
    TableMetadata,
    DatabaseSchema
)
from functions.llm_service import generate_verbose_descriptions

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

    for table_name, info in db_metadata.items():
        # Fetch descriptions from LLM
        print(f"LLM is generating description for table: {table_name}")
        ai_desc = generate_verbose_descriptions(table_name, info)
        
        # Create a lookup dictionary mapping column names to their AI descriptions
        desc_lookup = {col.name: col.description for col in ai_desc.columns}
        synonyms_lookup = {col.name: col.synonyms for col in ai_desc.columns}

        # Map columns to Pydantic objects with AI descriptions
        processed_columns = []
        for col in info.get('columns', []):
            col_name = col['name']
            
            processed_columns.append(ColumnMetadata(
                name=col_name,
                type=col['type'],
                nullable=col['nullable'],
                description=desc_lookup.get(col_name, "No description generated."),
                synonyms=synonyms_lookup.get(col_name, "None")
            ))

        # Map relationships (with a fallback to an empty list if none exist)
        processed_relationships = [
            RelationshipMetadata(**rel) for rel in info.get('relationships', [])
        ]

        # Create Table Object
        final_database_schema[table_name] = TableMetadata(
            name=table_name,
            description=ai_desc.description,
            synonyms=ai_desc.synonyms,
            columns=processed_columns,
            primary_key=info.get('primary_key', []),
            relationships=processed_relationships
        )

    # Wrap in the final container
    return DatabaseSchema(tables=final_database_schema)

def export_enriched_schema_to_json(schema: DatabaseSchema, filepath: str = "enriched_schema.json") -> str:
    """
    Helper function to export the Pydantic DatabaseSchema object to a JSON file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(schema.model_dump_json(indent=2))
    
    return filepath