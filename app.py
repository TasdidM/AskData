import uuid
import streamlit as st
import traceback
from agent.graph import app  # Imports your compiled LangGraph application

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Database Assistant Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Database Assistant Agent")
st.caption("Chat with your database using natural language and watch the agent build SQL queries.")

# --- SESSION STATE INITIALIZATION ---
# 1. Initialize a unique thread ID for tracking state across user runs
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 2. Maintain a history of the chat messages to render in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bundle the session configuration for LangGraph
session_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# --- SIDEBAR INFO ---
with st.sidebar:
    st.subheader("Session Metadata")
    st.text_input("Active Thread ID", value=st.session_state.thread_id, disabled=True)
    if st.button("Clear Chat / Reset Session", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# --- RENDER CHAT HISTORY ---
# Display previous messages on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg and msg["sql"]:
            st.markdown(f"**Active Table Cache:** `{msg['active_table']}`")
            st.code(msg["sql"], language="sql")

# --- USER INPUT & AGENT EXECUTION ---
if user_input := st.chat_input("Ask something about the database..."):
    
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Store user message in history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process the message with LangGraph
    with st.chat_message("assistant"):
        try:
            inputs = {"query": user_input}
            
            # Using Streamlit's collapsible status container to simulate node logging
            with st.status("🤖 Agent thinking and processing nodes...", expanded=True) as status:
                for output in app.stream(inputs, session_config, stream_mode="updates"):
                    for node_name, node_output in output.items():
                        st.write(f"⚙️ **[Node Finished]:** `{node_name}`")
                
                status.update(label="✅ Graph Execution Completed!", state="complete", expanded=False)
            
            # Fetch the final state values
            final_state = app.get_state(session_config).values
            generated_sql = final_state.get("generated_sql", "No SQL generated.")
            active_table = final_state.get("active_table_name", "None")
            
            # Formatted AI response presentation
            ai_response = "Here is the generated query for your request:"
            st.markdown(ai_response)
            st.markdown(f"**Active Table Cache:** `{active_table}`")
            st.code(generated_sql, language="sql")
            
            # Append complete response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "sql": generated_sql,
                "active_table": active_table
            })
            
            # Force instant app refresh to clear active execution elements and fix shadowing/duplication
            st.rerun()
            
        except Exception as e:
            st.error("💥 An error occurred during graph execution.")
            with st.expander("Full Crash Log"):
                st.code(traceback.format_exc(), language="python")