from typing import Literal
from agent.state import AssistantState

def route_after_intent(state: AssistantState) -> Literal["get_new_table_prompt_node", "get_comparison_table_prompt_node", "get_same_table_prompt_node"]:
    """Determines whether we need to hit the vector database or skip it entirely."""
    if state["intent"] == "NEW_TABLE":
        return "get_new_table_prompt_node" 
    else:
        return "get_same_table_prompt_node"   