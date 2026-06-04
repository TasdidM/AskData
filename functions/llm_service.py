from functions.llm_pydantic import TableDescription
from typing import Dict, Any
import ollama

def generate_verbose_descriptions(table_name: str, table_metadata: Dict[str, Any]):
    """
    Generates verbose business documentation for a table using its full technical schema.
    
    :param table_name: Name of the SQL table
    :param table_metadata: Dictionary containing columns, primary_key, and relationships
    """
    
    # 1. Format columns into a readable string for the prompt
    columns_formatted = ""
    for col in table_metadata.get('columns', []):
        nullable_str = "NULL" if col.get('nullable') else "NOT NULL"
        columns_formatted += f"  - {col['name']} ({col['type']}) {nullable_str}\n"
    
    # 2. Format primary keys
    pk_formatted = ", ".join(table_metadata.get('primary_key', []))
    
    # 3. Format relationships (Foreign Keys)
    rel_formatted = ""
    relationships = table_metadata.get('relationships', [])
    if relationships:
        for rel in relationships:
            from_cols = ", ".join(rel['from_column'])
            to_cols = ", ".join(rel['to_column'])
            rel_formatted += f"  - FK: ({from_cols}) references {rel['to_table']}({to_cols})\n"
    else:
        rel_formatted = "  - None"

    # 4. Construct the comprehensive prompt
    prompt = f"""
    Generate professional, verbose business documentation for the following SQL structure:
    
    TABLE NAME: {table_name}
    
    COLUMNS:
    {columns_formatted}
    PRIMARY KEY: 
      - ({pk_formatted})
      
    RELATIONSHIPS:
    {rel_formatted}
    
    SAMPLE DATA:
    {table_metadata.get("sample_data", [])}

    Your task is to:
    1. Infer and describe the overall business purpose and domain of the table.
    2. You should look at the provided data and try to understand what each table are trying to contain using the SAMPLE DATA.
    2. Describe the business purpose, intent, and alternative business terms (synonyms) for each column, leveraging the context of data types and foreign key relationships to make descriptions highly accurate.
    
    Return ONLY the JSON matching the requested schema.
    """

    # 5. Call Ollama
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are an expert technical writer and data governance analyst. Output JSON only.'},
            {'role': 'user', 'content': prompt}
        ],
        format=TableDescription.model_json_schema(),
        options={'temperature': 0.3} 
    )

    return TableDescription.model_validate_json(response['message']['content'])


