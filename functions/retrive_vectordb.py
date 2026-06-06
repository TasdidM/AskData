import logging
from typing import List, Tuple, Dict, Any, Optional
from pymongo.errors import PyMongoError

# Set up a logger for this module
logger = logging.getLogger(__name__)

def find_top_matches(
    collection: Any, 
    embedding_model: Any, 
    query_text: str, 
    vector_index_name: str,
    limit: int = 5,
    num_candidates: int = 100
) -> List[Dict[str, Any]]:
    """
    Performs an Atlas Vector Search on a MongoDB collection based on a text query.
    
    Args:
        collection: The PyMongo collection object.
        embedding_model: The model object used to generate embeddings (must implement `.embed()`).
        query_text (str): The raw text query from the user.
        vector_index_name (str): The name of the Atlas Vector Search index.
        limit (int): The maximum number of documents to return. Defaults to 5.
        num_candidates (int): The number of initial candidate matches to scan. Defaults to 100.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the matching documents and their scores.
    """
    logger.debug(f"Starting vector search for query: '{query_text}'")

    # 1. Generate the query vector
    try:
        # Assuming the embed() method returns a generator or iterator
        query_vector = next(embedding_model.embed(query_text)).tolist()
    except Exception as e:
        # logger.exception automatically includes the traceback for easier debugging
        logger.exception(f"Failed to generate embedding for query '{query_text}'")
        raise ValueError(f"Embedding generation failed for query: '{query_text}'") from e

    # 2. Define the Atlas Vector Search pipeline stage
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",          # Target field containing the vectors
                "queryVector": query_vector,  # The embedded user query
                "numCandidates": num_candidates, # Broad pool of initial matches to evaluate
                "limit": limit                # Final number of results to return
            }
        },
        {
            # 3. Shape the output data to return only necessary fields
            "$project": {
                "_id": 0,
                "table_name": 1,
                "content": 1,
                "score": {
                    "$meta": "vectorSearchScore"  # Extract the similarity score
                }
            }
        }
    ]

    # 4. Execute the aggregation pipeline
    try:
        cursor = collection.aggregate(pipeline)
        top_matches = list(cursor)
        
        logger.info(f"Vector search completed successfully. Found {len(top_matches)} matches.")
        return top_matches

    except PyMongoError as e:
        logger.error(f"MongoDB Vector Search aggregation failed: {e}")
        # Returning an empty list rather than crashing, though raising an error 
        # might be preferable depending on downstream requirements.
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred during search execution: {e}")
        return []
    

def format_search_results(top_matches: List[Dict[str, Any]]) -> Tuple[str, List[Optional[str]]]:
    """
    Formats the raw MongoDB vector search results into a single readable string 
    and extracts a list of associated table names.
    
    Args:
        top_matches (List[Dict[str, Any]]): The document matches returned from the pipeline.
        
    Returns:
        Tuple[str, List[Optional[str]]]: A tuple containing:
            - A formatted string of all retrieved content.
            - A list of table names extracted from the matching documents.
    """
    # Guard clause: handle empty results immediately
    if not top_matches:
        logger.info("No matches provided to formatter. Returning empty results.")
        return "", []

    logger.debug(f"Formatting {len(top_matches)} search results.")
    
    retrieval_content = ""
    primary_table_names = []
    
    # Enumerate starting at 1 to create human-readable indexes (e.g., [1], [2])
    for index, doc in enumerate(top_matches, start=1):
        
        # Safely grab the fields using .get() to prevent KeyErrors if data is missing
        table_name = doc.get("table_name")
        content = doc.get("content", "No content available")
        score = doc.get("score", 0.0)
        
        logger.debug(f"Processing Match {index} | Table: {table_name} | Score: {score:.4f}")
        
        primary_table_names.append(table_name)
        
        # Append the formatted string with a distinct boundary
        retrieval_content += f"[{index}] {content}\n"
        
    return retrieval_content, primary_table_names