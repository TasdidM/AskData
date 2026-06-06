import logging
from typing import Dict, Any
import ollama
from pydantic import ValidationError

from metadata_builder.LLM_pydantic import TableDescription

# Configure a logger for this module. 
# In a development environment, set the logging level to DEBUG to see the prompts and responses.
logger = logging.getLogger(__name__)

def _build_documentation_prompt(table_name: str, table_metadata: Dict[str, Any]) -> str:
    """
    Helper function to construct the prompt for the LLM.
    Isolating this logic makes it easier to unit test the prompt generation independently.
    """
    # 1. Format columns
    columns = table_metadata.get('columns', [])
    columns_formatted = "\n".join(
        f"  - {col.get('name')} ({col.get('type')}) {'NULL' if col.get('nullable') else 'NOT NULL'}"
        for col in columns
    )

    # 2. Format primary keys
    pk_formatted = ", ".join(table_metadata.get('primary_key', []))

    # 3. Format relationships (Foreign Keys)
    relationships = table_metadata.get('relationships', [])
    if relationships:
        rel_formatted = "\n".join(
            f"  - FK: ({', '.join(rel.get('from_column', []))}) references {rel.get('to_table')}({', '.join(rel.get('to_column', []))})"
            for rel in relationships
        )
    else:
        rel_formatted = "  - None"

    # 4. Extract sample data gracefully
    sample_data = table_metadata.get("sample_data", [])

    # 5. Construct and return the final prompt
    return f"""
    Generate professional, verbose business documentation for the following SQL structure:
    
    TABLE NAME: {table_name}
    
    COLUMNS:
    {columns_formatted}
    
    PRIMARY KEY: 
      - ({pk_formatted})
      
    RELATIONSHIPS:
    {rel_formatted}
    
    SAMPLE DATA:
    {sample_data}

    Your task is to:
    1. Infer and describe the overall business purpose and domain of the table.
    2. Look at the provided SAMPLE DATA to understand what context the table is trying to capture.
    3. Describe the business purpose, intent, and alternative business terms (synonyms) for each column, leveraging the context of data types and foreign key relationships to make descriptions highly accurate.
    
    Return ONLY the JSON matching the requested schema.
    """


def generate_verbose_descriptions(table_name: str, table_metadata: Dict[str, Any]) -> TableDescription:
    """
    Generates verbose business documentation for a table using its full technical schema.
    
    :param table_name: Name of the SQL table.
    :param table_metadata: Dictionary containing columns, primary_key, relationships, and sample_data.
    :return: A parsed TableDescription Pydantic model.
    :raises Exception: If the LLM call or JSON validation fails.
    """
    
    # Build the prompt using our helper function
    prompt = _build_documentation_prompt(table_name, table_metadata)
    
    # Log the prompt at DEBUG level so developers can inspect exactly what is sent to the LLM
    logger.debug(f"Sending prompt to Ollama for table '{table_name}':\n{prompt}")

    try:
        # Execute the LLM call
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': 'You are an expert technical writer and data governance analyst. Output JSON only.'},
                {'role': 'user', 'content': prompt}
            ],
            format=TableDescription.model_json_schema(),
            options={'temperature': 0.3} 
        )
        
        raw_response_content = response['message']['content']
        
        # Log the raw response at DEBUG level to help troubleshoot hallucinated or malformed JSON
        logger.debug(f"Received raw response from Ollama for table '{table_name}':\n{raw_response_content}")

    except Exception as e:
        logger.error(f"Failed to communicate with Ollama API for table '{table_name}': {e}")
        raise

    try:
        # Validate and return the Pydantic model
        return TableDescription.model_validate_json(raw_response_content)
    
    except ValidationError as e:
        logger.error(f"Failed to validate JSON response for table '{table_name}'. "
                     f"Response content: {raw_response_content}")
        raise ValueError(f"LLM returned invalid JSON structure: {e}")