from functions.metadata_extraction import extract_db_metadata, generate_schema_from_db, export_schema_to_json
from functions.db_connection import CONNECTION_URL
import json
from functions.llm_service import generate_verbose_descriptions
from functions.schema_builder import build_enriched_schema, export_enriched_schema_to_json

# basic schema metadata extraction
schema_dict = generate_schema_from_db(CONNECTION_URL)
db_base_schema_path = export_schema_to_json(schema_dict)
print(f"Database base schema saved to filepath: {db_base_schema_path}")


# enriched schema metadata extraction
# metadata = extract_db_metadata(CONNECTION_URL)
# schema = build_enriched_schema(metadata)
# db_enriched_schema_path = export_enriched_schema_to_json(schema)
# print(f"Database enriched schema saved to filepath: {db_enriched_schema_path}")




# for table_name, info in metadata.items():
#     ai_desc = generate_verbose_descriptions(table_name, info)
#     print(ai_desc)
#     break


