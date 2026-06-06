from typing import TypedDict, Literal, Optional, List, Dict

# Define the Shared State Schema
class AssistantState(TypedDict):
    query: str
    active_table_name: Optional[List[str]]
    active_table_schema: Optional[List[Dict]]
    intent: Optional[Literal["SAME_TABLE", "NEW_TABLE"]]
    chat_history: List[Dict[str, str]]
    user_prompt: Optional[str]
    top_p_tables: Optional[List[str]]
    generated_sql: Optional[str]

