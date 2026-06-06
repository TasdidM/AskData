import json
import logging
from typing import Tuple, Optional, List, Dict, Any

import ollama
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema
from sqlglot.errors import OptimizeError, ParseError

# Configure standard logging for better debugging and trace generation
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def _validate_sql(sql_query: str, schema: MappingSchema, dialect: str) -> Tuple[bool, Optional[str]]:
    """
    Helper function to validate SQL syntax, security, and schema.
    Returns a tuple of (is_valid, error_message).
    """
    try:
        expression = parse_one(sql_query, read=dialect)
        
        # 1. Safety check: Enforce read-only SELECT queries
        # If the query uses CTEs (WITH), we need to unwrap it to check the main query
        unwrapped_expr = expression.this if isinstance(expression, exp.With) else expression
        if not isinstance(unwrapped_expr, (exp.Select, exp.Union)):
            return False, "Security Policy Violation: Only read-only SELECT statements are allowed."
        
        # 2. Structural check: Validate tables & columns exist in the schema
        qualify(expression, schema=schema)
        
        return True, None
        
    except (ParseError, OptimizeError, ValueError) as e:
        return False, f"SQL Validation Error: {str(e)}"
    except Exception as e:
        logger.error("Unexpected error during SQLGlot validation", exc_info=True)
        return False, f"System Error during validation: {str(e)}"

def get_validated_sql(
    user_prompt: str, 
    system_prompt: str, 
    database_schema: Dict[str, Any], 
    dialect: str = "mysql", 
    llm_model: str = "llama3", 
    max_attempts: int = 3
) -> Tuple[Optional[str], str]:
    """
    Generates a SQL query using an LLM and validates it against a provided schema.
    If validation fails, prompts the LLM to self-correct using the error output.
    
    Returns:
        Tuple containing (extracted_sql_string, status_message).
        If all attempts fail, extracted_sql_string will be None.
    """
    current_prompt = user_prompt
    mapped_schema = MappingSchema(database_schema, dialect=dialect)

    for attempt in range(1, max_attempts + 1):
        logger.info(f"🔄 Attempt {attempt}/{max_attempts}: Generating SQL...")
        
        try:
            # 1. Generate SQL via LLM
            response = ollama.generate(
                model=llm_model,
                system=system_prompt,
                prompt=current_prompt,
                format="json",
                options={"temperature": 0.0}  # Deterministic output
            )
            
            # 2. Extract JSON payload
            raw_output = response.get('response', '')
            parsed_json = json.loads(raw_output)
            extracted_sql = parsed_json.get("sql", "").strip()
            
            logger.debug(f"LLM Output SQL:\n{extracted_sql}")
            
            if not extracted_sql:
                raise ValueError("The JSON object was missing the 'sql' key or the query was empty.")
            
            # 3. Validate the extracted SQL using our helper function
            is_valid, error_message = _validate_sql(extracted_sql, mapped_schema, dialect)
            
            if is_valid:
                logger.info(f"✅ Success on attempt {attempt}!")
                return extracted_sql, f"Successfully validated after {attempt} attempt(s)."
            else:
                # Raise the error so it gets caught and triggers the self-correction loop
                raise ValueError(error_message)
                
        except (json.JSONDecodeError, ValueError) as e:
            error_details = str(e)
            logger.warning(f"❌ Attempt {attempt} failed: {error_details}")
            
            # Trigger self-correction if we haven't maxed out attempts
            if attempt < max_attempts:
                # Dedenting the multiline string avoids injecting unnecessary whitespace into the prompt
                current_prompt = (
                    "You have run into an unexpected error.\n\n"
                    f"Previous Context:\n{user_prompt}\n\n"
                    f"Your previous SQL generation failed database validation constraints. "
                    f"You have the following database Schema:\n{database_schema}\n\n"
                    "Instruction:\n"
                    "Review the allowed Database Schema above, analyze the specific error details below, "
                    f"fix the column/syntax mismatch, and regenerate the correct {dialect} query.\n\n"
                    f"**Error Spotted:** {error_details}\n\n"
                    "Provide your corrected response as the requested JSON object below:"
                )
            else:
                logger.error(f"Failed to generate valid SQL after {max_attempts} attempts.")
                return None, f"Failed after {max_attempts} attempts. Last error: {error_details}"

def extract_table_from_sql(sql_query: str) -> List[str]:
    """
    Parses a SQL query and extracts actual table names, filtering out temporary CTEs.
    """
    try:
        parsed = parse_one(sql_query)

        # 1. Gather all Common Table Expression (CTE) aliases to exclude them later
        cte_names = {cte.alias_or_name for cte in parsed.find_all(exp.CTE)}

        # 2. Extract real tables (ignoring our CTE names)
        real_tables = list({
            table.name for table in parsed.find_all(exp.Table) 
            if table.name not in cte_names
        })

        logger.debug(f"Extracted real tables: {real_tables}")
        return real_tables

    except Exception as e:
        logger.error(f"Failed to extract tables from SQL: {e}")
        return []