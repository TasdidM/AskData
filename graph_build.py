from functions.graph_builder import build_knowledge_graph, export_interactive_visualization
import pickle

filepath = "enriched_schema.json"
with open(filepath, 'r') as schema_json:
    # Generate the Graph
    kg = build_knowledge_graph(schema_json)

# Debug Summary Profiler
print(f"Knowledge Graph Generation Statistics:")
print(f"--------------------------------------")
print(f"Total Nodes Processed: {kg.number_of_nodes()}")
print(f"Total Edges Generated: {kg.number_of_edges()}")

# Showcase sample metadata retrieval format used by a RAG Pipeline
print("\nSample RAG Metadata Output (From 'track' table):")
print(f" - Node Attributes: {kg.nodes['track.AlbumId']}")

# Export Visualization HTML
export_interactive_visualization(kg)

# Save to a pickle file
with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump(kg, f)

print("Saved knowledge graph to 'knowledge_graph.pkl'")