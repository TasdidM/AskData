import streamlit as st
import ollama
import pickle
import os
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Import your custom functions
from functions.graph_builder import generate_rag_prompt, get_filtered_subgraphs, get_subgraphs_joins
from functions.create_vectordb import connect_mongodb_collection
from functions.retrive_vectordb import find_top_matches, format_search_results

# --- Page Config ---
st.set_page_config(page_title="GraphRAG to SQL", page_icon="📊", layout="centered")

# --- Caching Heavy Resources ---
# We cache these so they don't reload every time the user types a new query
@st.cache_resource(show_spinner="Loading Models and Knowledge Graph...")
def initialize_system():
    load_dotenv()
    
    # Load Environment Variables
    env_vars = {
        "mongo_uri": os.getenv("MONGO_URI"),
        "db_name": os.getenv("VECTOR_DB_NAME"),
        "collection_name": os.getenv("COLLECTION_NAME"),
        "vector_index_name": os.getenv("VECTOR_INDEX_NAME"),
        "llm_model": os.getenv("LLM_MODEL")
    }
    
    # Load Models & DB
    embedding_model = TextEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))
    mongodb_collection = connect_mongodb_collection(
        env_vars["mongo_uri"], 
        env_vars["db_name"], 
        env_vars["collection_name"]
    )
    
    # Load Knowledge Graph Pickle
    with open("knowledge_graph.pkl", "rb") as f:
        kg_loaded = pickle.load(f)
        
    return env_vars, embedding_model, mongodb_collection, kg_loaded

# Initialize the resources
try:
    env, embedding_model, mongodb_collection, kg_loaded = initialize_system()
except Exception as e:
    st.error(f"Failed to initialize system: {e}")
    st.stop()


# --- Main App Interface ---
st.title("📊 GraphRAG Text-to-SQL")
st.markdown("Ask a question in plain English, and the system will use your Knowledge Graph and MongoDB Vector Database to generate the exact SQL code.")

user_query = st.text_input(
    "What do you want to know?", 
    placeholder="e.g., Which employee generated the most revenue in 2025?"
)

if st.button("Generate SQL", type="primary"):
    if not user_query:
        st.warning("Please enter a question first.")
    else:
        try:
            # --- Step 1: Retrieval (Vector DB + Knowledge Graph) ---
            with st.status("🔍 Querying Database and Knowledge Graph...", expanded=True) as status:
                st.write("Searching Vector Database...")
                top_matches = find_top_matches(
                    mongodb_collection, 
                    embedding_model, 
                    user_query, 
                    env["vector_index_name"]
                )
                
                st.write("Extracting Subgraphs...")
                vector_content, table_names = format_search_results(top_matches)
                subgraphs = get_filtered_subgraphs(kg_loaded, table_names)
                graph_content = get_subgraphs_joins(subgraphs)
                
                extracted_context = f"{vector_content} \n {graph_content}"
                status.update(label="Context retrieved successfully!", state="complete", expanded=False)

            # --- Step 2: Prompt Generation ---
            final_prompt_string = generate_rag_prompt(user_query, extracted_context)
            
            # (Optional) Allow the user to inspect what goes into the LLM
            with st.expander("🛠️ View Retrieved Context & LLM Prompt"):
                st.subheader("Extracted Context")
                st.text(extracted_context)
                st.subheader("Final Prompt to Ollama")
                st.text(final_prompt_string)

            # --- Step 3: LLM Generation ---
            with st.spinner(f"Drafting SQL with Ollama ({env['llm_model']})..."):
                response = ollama.generate(
                    model=env["llm_model"], 
                    prompt=final_prompt_string, 
                    options={"temperature": 0.0}
                )
                
            # Clean and display output
            sql_output = response["response"].strip()
            if sql_output.startswith("```sql"):
                sql_output = sql_output[6:]
            elif sql_output.startswith("```"):
                sql_output = sql_output[3:]
            if sql_output.endswith("```"):
                sql_output = sql_output[:-3]

            st.success("Query generated successfully!")
            st.code(sql_output.strip(), language="sql")

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")