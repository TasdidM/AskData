from knowledge_graph_builder.graph_operations import get_filtered_subgraphs, get_subgraphs_joins
from vector_db_builder.create_vector_db import connect_mongodb_collection
from functions.retrive_vectordb import find_top_matches, format_search_results
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from fastembed import TextEmbedding
import ollama
import pickle
import os
from dotenv import load_dotenv
load_dotenv()



# Global configuration placeholders (Update with your actual configurations)
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("VECTOR_DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
vector_index_name = os.getenv("VECTOR_INDEX_NAME")

embedding_model = TextEmbedding(model_name = os.getenv("EMBEDDING_MODEL"))
llm_model = os.getenv("LLM_MODEL")


# ----------------------------------------------------
# 1. Intent Classification Components
# ----------------------------------------------------
class IntentClassification(BaseModel):
    intent: Literal["SAME_TABLE", "NEW_TABLE"] = Field(
        description="SAME_TABLE if the user query requires the current table. NEW_TABLE if they want a different table or topic."
    )
    reasoning: str = Field(
        description="A brief one-sentence breakdown of why you chose this intent."
    )


def call_intent_llm(query: str, chat_history: List[Dict[str, str]], active_table: str, llm_model: str = "llama3") -> str:
    """Uses a local Ollama model with a strict JSON schema to classify query intent 
    into one of three buckets: SAME_TABLE, COMPARISON_TABLE, or NEW_TABLE.
    """
    
    # Update the system prompt with explicit rules for the 3 intents
    system_prompt = f"""You are a strict query router for a database assistant.
    The user is currently examining a database table called: '{active_table}'.

    Analyze the dialogue history and the user's latest question. You must classify their intent into exactly one of these categories:

    1. SAME_TABLE:
    - Select this if the query is a follow-up about the active table. 
    - Examples: filtering, sorting, aggregations, or asking deep-dive questions using data solely within '{active_table}'.

    2. NEW_TABLE:
    - Select this if the query completely abandons '{active_table}' and switches to an entirely different, unrelated topic. The active table is no longer relevant to answering this new question.
    """
    
    # 3. Compile the messages thread (keeping the last 4 turns to prevent bloating)
    messages = [{"role": "system", "content": system_prompt}]
    
    for turn in chat_history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
        
    # Append the newest user query
    messages.append({"role": "user", "content": query})
    
    try:
        # 4. Fire the request to Ollama
        # Using a low temperature is crucial for deterministic categorization
        response = ollama.chat(
            model=llm_model,  # or qwen2.5:3b / llama3.2
            messages=messages,
            format=IntentClassification.model_json_schema(), # Enforces the strict structural schema
            options={"temperature": 0.0}                     # Crucial for deterministic classification
        )
        
        # 5. Parse and validate the response
        structured_data = IntentClassification.model_validate_json(response.message.content)
        print(f"🤖 Local LLM Reason: {structured_data.reasoning} \n Intent: {structured_data.intent}")
        return structured_data.intent

    except Exception as e:
        print(f"⚠️ Intent parsing failed: {e}. Defaulting to 'NEW_TABLE' for safety.")
        return "NEW_TABLE"
    

# ----------------------------------------------------
# 2. Database Extraction Components
# ----------------------------------------------------
def get_tables_by_name(table_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves the exact table metadata documents from MongoDB 
    matching a list of specific table names.
    """
    # 1. Build the query. 
    # Change "table_name" to whatever your field is called in MongoDB (e.g., "name")
    query = {"table_name": {"$in": table_names}}
    
    # 2. Query the collection
    cursor = connect_mongodb_collection(mongo_uri, db_name, collection_name).find(query)
    
    # 3. Cast the cursor results to a standard Python list
    results = list(cursor)
    
    return results

def new_table_content_extraction(user_query:str) -> str:

    # To load the Pickle back:
    with open("assets/knowledge_graph.pkl", "rb") as f:
        kg_loaded = pickle.load(f)

    # connect to the vectorDB
    mongodb_collection = connect_mongodb_collection(mongo_uri, db_name, collection_name)
    # find the top 5 matches using vector search 
    top_matches = find_top_matches(mongodb_collection, embedding_model,user_query, vector_index_name)
    # format the results 
    vector_content, table_names = format_search_results(top_matches)
    # find the raltional subgraphs between the tables
    subgraphs = get_filtered_subgraphs(kg_loaded, table_names)
    # extract content from the graphDB
    graph_content = get_subgraphs_joins(subgraphs)

    # reformate the whole context
    extracted_context = f"{vector_content} \n {graph_content}"
    return extracted_context, table_names


def same_table_content_extraction(active_table_list: list) -> str:
    # To load the Pickle back:
    with open("assets/knowledge_graph.pkl", "rb") as f:
        kg_loaded = pickle.load(f)

    vector_content = ""
    tables_content = get_tables_by_name(active_table_list)
    for table_content in tables_content:
        vector_content += table_content["content"]
    # find the raltional subgraphs between the tables
    subgraphs = get_filtered_subgraphs(kg_loaded, active_table_list)
    # extract content from the graphDB
    graph_content = get_subgraphs_joins(subgraphs)
    extracted_context = f"{vector_content} \n {graph_content}"
    return extracted_context

# if __name__ == "__main__":
#     t = same_table_context_extraction(["album", "customer"])
#     print(t)
    