
from functions.graph_builder import generate_SQL_rag_prompt, get_filtered_subgraphs, get_subgraphs_joins
from functions.create_vectordb import connect_mongodb_collection
from functions.retrive_vectordb import find_top_matches, format_search_results
from functions.sql_validation import get_validated_sql
from functions.metadata_extraction import generate_schema_from_db
import ollama
from sqlglot.schema import MappingSchema
import pickle
import os
from fastembed import TextEmbedding
from dotenv import load_dotenv
import json
load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("VECTOR_DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
vector_index_name = os.getenv("VECTOR_INDEX_NAME")

embedding_model = TextEmbedding(model_name = os.getenv("EMBEDDING_MODEL"))
llm_model = os.getenv("LLM_MODEL")

# To load the Pickle back:
with open("knowledge_graph.pkl", "rb") as f:
    kg_loaded = pickle.load(f)

print("\n Hi! I am your SQL generator, what do you want to know about Chinook Database?")
#  eg. 'Which Top 3 employee generated the most revenue in 2025?'
user_query = input("QUERY: ")


mongodb_collection = connect_mongodb_collection(mongo_uri, db_name, collection_name)
top_matches = find_top_matches(mongodb_collection, embedding_model,user_query, vector_index_name)
vector_content, table_names = format_search_results(top_matches)
subgraphs = get_filtered_subgraphs(kg_loaded, table_names)
graph_content = get_subgraphs_joins(subgraphs)

extracted_context = f"{vector_content} \n {graph_content}"
print(extracted_context)

# Step 2: Pass output directly into YOUR target function
system_prompt_string, user_prompt_string = generate_SQL_rag_prompt(user_query, extracted_context)


# Previewing what the prompt looks like right before going to Ollama
# print("--- Generated Prompt Payload Passed to LLM ---\n")
# print(final_prompt_string)
# print("\n-----------------------------------------------\n")

# # Step 3: Call Ollama to generate the final SQL code query
# print("Slicing data with Ollama...")
# response = ollama.generate(
#     model=llm_model, prompt=final_prompt_string, options={"temperature": 0.0}
# )



with open("db_base_schema.json", 'r') as file:
    database_schema = MappingSchema(json.load(file))
extract, text = get_validated_sql(user_prompt_string, system_prompt_string, database_schema)

print(extract)
print(text)










# from rich.console import Console

# console = Console()

# # Use rich's status indicator with a built-in spinner animation
# with console.status("[bold green]LLM is thinking and generating code...", spinner="dots"):
#     response = ollama.generate(
#         model=llm_model,
#         system=system_prompt_string,
#         prompt=user_prompt_string,
#         format="json",
#         options={"temperature": 0.0}
#     )

# # Everything inside the "with" block keeps the spinner alive. 
# # Once it exits, the spinner disappears and you can print the output:
# console.print("\n[bold blue]Generation Complete![/bold blue]")
# print(response['response'])

# with open("sql_generated_by_LLM", 'w', encoding='utf-8') as f:
#         f.write(response.model_dump_json(indent=2))

# print("\n### Final LLM Generated SQL Output ###")
# print(response["response"])