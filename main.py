# main.py
import uuid
from agent.graph import app  # Imports the compiled LangGraph application
import traceback
def run_interactive_agent():
    print("🤖 Database Assistant Agent Initialized!")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    # 1. Generate a unique thread ID for this specific chat session.
    # This is what LangGraph uses to track your chat history and active table state.
    session_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }
    
    # 2. Start the conversation loop
    while True:
        
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # 3. Prepare the input payload matching your AssistantState
            inputs = {"query": user_input}
            
            # 4. Invoke the graph. LangGraph handles the state updates behind the scenes.
            # We use .stream() instead of .invoke() so we can print out which node is currently executing!
            print("\n--- Execution Started ---")
            for output in app.stream(inputs, session_config, stream_mode="updates"):
                # This prints the name of the node that just finished running
                for node_name, node_output in output.items():
                    print(f"⚙️ [Node Finished]: {node_name}")
            print("--- Execution Finished ---\n")
            
            # 5. Fetch the final state to print the generated SQL
            final_state = app.get_state(session_config).values
            generated_sql = final_state.get("generated_sql", "No SQL generated.")
            active_table = final_state.get("active_table_name", "None")
            
            print(f"📊 Active Table Cache: {active_table}")
            print(f"💻 Generated SQL:\n{generated_sql}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nSession aborted.")
            break
        except Exception as e:
            print("\n💥 --- FULL CRASH LOG --- 💥")
            traceback.print_exc()  # <-- This will print the exact file and line number!
            print("💥 ---------------------- 💥")
            break

if __name__ == "__main__":
    run_interactive_agent()