from typing import List, Dict, Optional

def generate_system_prompt(sql_service_name: str = "MySQL") -> str:
    """
    Generates the system prompt instructing the LLM on its role and output format.
    
    Args:
        sql_service_name (str): The specific SQL dialect to generate (e.g., MySQL, PostgreSQL).
        
    Returns:
        str: The fully formatted system prompt.
    """
    system_prompt = f"""You are a Senior Data Engineer who is a professional in SQL writing in {sql_service_name}. 
    Your ONLY task is to translate natural language questions into a valid {sql_service_name} query wrapped inside a JSON object.

    CRITICAL FORMATTING RULES:
    1. Output a valid JSON object with exactly one key: "sql".
    2. The value of "sql" must be a single string containing the raw {sql_service_name} query.
    3. DO NOT include markdown syntax (like ```sql or ```) inside or outside the JSON.
    4. JSON STRING ESCAPING: To prevent JSON parsing errors, always use single quotes (') for string literals inside your SQL query (e.g., WHERE name = 'Alice'). Do not use unescaped double quotes.
    5. No commentary, no explanations. Output ONLY the JSON object.

    Example Output Format:
    {{
        "sql": "SELECT name FROM users WHERE id = 1 AND status = 'active';"
    }}
    """
    # Using textwrap.dedent (optional) can help format multi-line strings, 
    # but maintaining your exact indentation for safety.
    return system_prompt

def generate_rag_user_prompt(
    user_query: str, 
    extracted_context: str, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Combines dynamically retrieved graph context and conversational history 
    into a strict prompt for the LLM.
    
    Args:
        user_query (str): The latest question from the user.
        extracted_context (str): Schema context (subgraphs/joins) for the active tables.
        chat_history (Optional[List[Dict[str, str]]]): A list of dicts detailing previous turns,
            e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
    Returns:
        str: The fully assembled user prompt.
    """
    
    # 1. Format the conversation history into a readable string block
    history_str = ""
    if chat_history:
        history_str = "\nConversation History:\n"
        for message in chat_history:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = message.get("content", "")
            history_str += f"{role}: {content}\n"
    
    # 2. Inject the history block and schema right before the final query
    user_prompt = f"""The following are the extracted possible relevant tables with their columns name, with subgraph of relations between the tables:
        {extracted_context}

        Instructions:
        Based ONLY on the schema context provided above, generate a valid query to answer the user's question. Do not guess or hallucinate table names or columns. Use exact names provided for the tables and columns.
        {history_str}
        Current User Question: {user_query}

        Provide the response as the requested JSON object below:"""

    return user_prompt