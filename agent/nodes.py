from LLM_prompts.prompt_generation import generate_rag_user_prompt, generate_system_prompt
from functions.sql_validation import get_validated_sql, extract_table_from_sql
from agent.utils import call_intent_llm, new_table_content_extraction, same_table_content_extraction
from agent.state import AssistantState
from typing import Dict, Any
import json


# Define the Nodes (The execution stations)
def intent_analyzer_node(state: AssistantState) -> Dict[str, Any]:
    """PHASE 2 GATING: Runs intent classifier ONLY if a table is already active."""
    
    active_table = state.get("active_table_name")
    
    # Turn 1: No active table yet, skip intent checking and flag as NEW_TABLE
    if not active_table:
        return {"intent": "NEW_TABLE"}

    
    # Mocking successful LLM response for demonstration:
    response = call_intent_llm(
        query=state["query"], 
        chat_history=state["chat_history"], 
        active_table=active_table
    )
    
    # response.intent will be exactly "NEW_TABLE" or "SAME_TABLE"
    return {"intent": response}


def get_new_table_prompt_node(state: AssistantState):
    """Generates prompts based on new tables"""

    print("Generating prompt for the new tables...")
    content, table_names =  new_table_content_extraction(state["query"])
    # you need to system it another away####
    chat_history = state.get("chat_history", [])[-4:]
    user_prompt_string = generate_rag_user_prompt(state["query"], content, chat_history)
    return {
        "user_prompt": user_prompt_string,
        "top_p_tables": table_names
    }


def get_same_table_prompt_node(state: AssistantState):
    """Generates prompt for the same tables"""

    print(f"Generating prompt for table: {state['active_table_name']}")

    content =  same_table_content_extraction(state["active_table_name"])
    chat_history = state.get("chat_history", [])[-4:]
    user_prompt_string = generate_rag_user_prompt(state["query"], content, chat_history)
    return {
        "user_prompt": user_prompt_string
    }


def llm_generation_and_validation_node(state: AssistantState):
    with open("assets/db_base_schema.json", 'r') as file:
        raw_schema_dict = json.load(file)
    target_tables = state.get("top_p_tables", [])
    
    target_tables_lower = [t.lower() for t in target_tables]

    filtered_database_schema = {
        name: data for name, data in raw_schema_dict.items() 
        if name.lower() in target_tables_lower
    }

    system_prompt = generate_system_prompt(sql_service_name = "MySQL")
    extracted_sql, _ = get_validated_sql(state["user_prompt"], system_prompt, filtered_database_schema)

    active_tables = extract_table_from_sql(extracted_sql)
    active_tables_lower = [t.lower() for t in active_tables]

    active_table_schema = {
        name: data for name, data in raw_schema_dict.items() 
        if name.lower() in active_tables_lower
    }


    new_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": extracted_sql}
    ]

    return {
        "generated_sql": extracted_sql, 
        "chat_history": new_history,
        "active_table_name": active_tables,
        "active_table_schema": active_table_schema
    }
