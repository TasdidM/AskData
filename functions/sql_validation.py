import json
import ollama
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema
from sqlglot.errors import OptimizeError, ParseError
from typing import Tuple, Optional

# Sqlglot Validation Function (with the LLM error builder)
def validate_sql_against_schema(sql_query: str, schema: MappingSchema, dialect: str = "postgres"):
    try:
        expression = parse_one(sql_query, read=dialect)
        
        # Strictly enforce SELECT statements
        unwrapped = expression.this if isinstance(expression, exp.With) else expression
        if not isinstance(unwrapped, (exp.Select, exp.Union)):
            raise ValueError("Non-SELECT statement detected. Only read-only SELECT queries are allowed.")
        
        # Qualify columns against the schema
        qualify(expression, schema=schema)
        return True, "SQL is valid and matches schema."
        
    except ParseError as pe:
        return False, f"Syntax Error: {pe}"
    except OptimizeError as oe:
        return False, f"Schema/Validation Error: {oe}"
    except Exception as e:
        return False, f"Error: {e}"
    
def get_validated_sql(
    user_prompt: str, 
    system_prompt: str, 
    database_schema: MappingSchema,
    dialect: str = "mysql",
    llm_model: str = "llama3", # Replace with your actual model
    max_attempts: int = 3
) -> Tuple[Optional[str], str]:
    """
    Generates a PostgreSQL query using an LLM and validates it against a schema.
    If validation fails, it loops back to the LLM with error feedback for self-correction.
    
    Returns:
        (extracted_sql, status_message)
        extracted_sql will be None if all attempts fail.
    """
    current_user_prompt = user_prompt

    # 3. Begin the Self-Correction Loop
    for attempt in range(1, max_attempts + 1):
        print(f"🔄 Attempt {attempt}/{max_attempts}: Generating SQL...")
        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=llm_model,
                system=system_prompt,
                prompt=current_user_prompt,
                format="json",
                options={"temperature": 0.0}  # Keep it deterministic
            )
            
            # Step A: Parse JSON Wrapper
            raw_output = response['response']
            parsed_json = json.loads(raw_output)
            extracted_sql = parsed_json.get("sql", "").strip()
            
            if not extracted_sql:
                raise ValueError("The JSON object was missing the 'sql' key or it was empty.")
            
            # Step B: Run through SQLGlot validation
            # (Inlined validation logic for self-containment)
            expression = parse_one(extracted_sql, read=dialect)
            
            # Safety check: enforce SELECT queries only
            unwrapped = expression.this if isinstance(expression, exp.With) else expression
            if not isinstance(unwrapped, (exp.Select, exp.Union)):
                raise ValueError("Security Policy Violation: Only read-only SELECT statements are allowed.")
            
            # Structural check: validate tables & columns
            qualify(expression, schema=database_schema)
            
            # If we reach this line, it passed everything!
            print(f"✅ Success on attempt {attempt}!")
            return extracted_sql, f"Successfully validated after {attempt} attempt(s)."
            
        except (json.JSONDecodeError, ParseError, OptimizeError, ValueError, Exception) as e:
            error_message = str(e)
            print(f"❌ Attempt {attempt} failed validation: {error_message}")
            
            # If we have attempts left, rewrite the prompt with explicit feedback
            if attempt < max_attempts:
                current_user_prompt = f"""### Previous Context
                {user_prompt}

                ### Instruction
                Your previous SQL generation failed validation constraints. Review the error details below, fix the issue, and regenerate the correct PostgreSQL query. 

                **Error Spotted:** {error_message}

                Provide the corrected response as the requested JSON object below:"""
            else:
                # Out of attempts
                return None, f"Failed to generate valid SQL after {max_attempts} attempts. Last error: {error_message}"