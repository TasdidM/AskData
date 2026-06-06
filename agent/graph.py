from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Import items from our local files
from agent.state import AssistantState
from agent.routers import route_after_intent
from agent.nodes import (
    intent_analyzer_node,
    get_new_table_prompt_node,
    get_same_table_prompt_node,
    llm_generation_and_validation_node
)

# 1. Wire the Graph Together
workflow = StateGraph(AssistantState)

# 2. Add all nodes with consistent string identifiers
workflow.add_node("intent_analyzer", intent_analyzer_node)
workflow.add_node("get_new_table_prompt_node", get_new_table_prompt_node)
workflow.add_node("get_same_table_prompt_node", get_same_table_prompt_node)
workflow.add_node("llm_generation_and_validation", llm_generation_and_validation_node)

# 3. Build the edges and conditional pathways
workflow.add_edge(START, "intent_analyzer")

# After intent analysis, conditionally route the flow
workflow.add_conditional_edges(
    "intent_analyzer",
    route_after_intent,
    {
        "get_new_table_prompt_node": "get_new_table_prompt_node",
        "get_same_table_prompt_node": "get_same_table_prompt_node"
    }
)

# Prompt builders both funnel into the validation/generation worker
workflow.add_edge("get_new_table_prompt_node", "llm_generation_and_validation")
workflow.add_edge("get_same_table_prompt_node", "llm_generation_and_validation")
workflow.add_edge("llm_generation_and_validation", END)

# 4. Compile with an in-memory checkpointer
app = workflow.compile(checkpointer=InMemorySaver())
