from pymongo.errors import PyMongoError

def find_top_matches(collection, embedding_model, query_text: str, vector_index_name: str) -> list:
    """
    Performs an Atlas Vector Search on a MongoDB collection based on a text query.
    
    Parameters:
    - collection: The PyMongo collection object.
    - embedding_model: The model object used to generate embeddings (must have an .embed() method).
    - query_text (str): The raw text query from the user.
    - vector_index_name (str): The name of the Atlas Vector Search index.
    
    Returns:
    - list: A list of the top 5 matching documents with their scores.
    """
    try:
        # 1. Generate the query vector
        # Using next() as per your original implementation
        query_vector = next(embedding_model.embed(query_text)).tolist()
    except Exception as e:
        print(f"Failed to generate embedding for query '{query_text}': {e}")
        raise ValueError("Embedding generation failed.") from e

    # 2. Define the Atlas Vector Search pipeline stage
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",          # The field where vectors are stored
                "queryVector": query_vector,  # The embedded user query
                "numCandidates": 100,         # Broad pool of initial matches
                "limit": 5                    # Strictly return the top 5 results
            }
        },
        {
            # 3. Shape the output data
            "$project": {
                "_id": 0,
                "table_name": 1,
                "content": 1,
                "score": {
                    "$meta": "vectorSearchScore" # Capture the similarity score
                }
            }
        }
    ]

    try:
        # 4. Execute the aggregation pipeline
        cursor = collection.aggregate(pipeline)
        
        # 5. Convert cursor to a standard Python list
        top_5_matches = list(cursor)
        return top_5_matches

    except PyMongoError as e:
        # Catch specific MongoDB errors (e.g., connection issues, wrong index name)
        print(f"MongoDB Vector Search aggregation failed: {e}")
        # Depending on your app, you can return an empty list or re-raise
        return []
    except Exception as e:
        # Catch any other unexpected python errors
        print(f"An unexpected error occurred during search: {e}")
        return []
    
def format_search_results(top_matches: list) -> tuple:
    """
    Formats the raw MongoDB vector search results into a readable string 
    and extracts the table name.
    
    Parameters:
    - top_matches (list): The list of document matches returned from the aggregation pipeline.
    
    Returns:
    - tuple: (retrieval_content_string, table_name)
    """
    # Guard clause in case the list is empty
    if not top_matches:
        print("\n--- No Results Found ---")
        return "", None

    print(f"\n--- Top {len(top_matches)} Results Found ---")
    
    retrieval_content = ""
    primary_table_name = []
    
    for index, doc in enumerate(top_matches, start=1):
        # Safely grab the fields using .get() to prevent KeyErrors
        table_name = doc.get("table_name")
        content = doc.get("content", "No content available")
        
        primary_table_name.append(table_name)
        
        # Append the formatted string
        retrieval_content += f"[{index}]{content}\n"
        
    # Print the final compiled string to the console as requested
    #print(retrieval_content)
    
    # Return both the formatted string and the table name for use in the next steps
    return retrieval_content, primary_table_name