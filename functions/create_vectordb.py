from fastembed import TextEmbedding
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ConnectionFailure, ConfigurationError
from pymongo.collection import Collection
import certifi
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")

# Ensure your parse function works with the schema structure
def parse_schema_to_documents(json_filepath):
    with open(json_filepath, 'r', encoding='utf-8') as file:
        schema_data = json.load(file)
        
    documents = []
    
    # Safely extract the tables object
    for table_name, table_info in schema_data.get('tables', {}).items():
        description = table_info.get('description', '')
        synonyms = table_info.get('synonyms', '')
        
        content = f"Table Name: {table_name}\n"
        content += f"Description: {description}\n"
        content += f"Synonyms: {synonyms}\n"
        content += "Columns:\n"
        
        for col in table_info.get('columns', []):
            content += f" - {col['name']} ({col['type']}): {col['description']} (Synonyms: {col.get('synonyms', '')})\n"
            
        documents.append({
            "table_name": table_name,
            "description": description,
            "content": content
        })
        
    return documents

def connect_mongodb_collection(mongo_uri:str, db_name:str, collection_name:str):
    try:
        print("Connecting to MongoDB...")
        # Added serverSelectionTimeoutMS to prevent indefinite hanging if the server is down
        client = MongoClient(
            mongo_uri, 
            tlsCAFile=certifi.where(), 
            serverSelectionTimeoutMS=5000
        )
        
        # PyMongo connects lazily. Pinging the admin database forces a real connection check.
        client.admin.command('ping')
        
        db = client[db_name]
        collection = db[collection_name]
        
        print(f"Successfully connected to '{db_name}.{collection_name}'!")
        return collection

    except ConnectionFailure as e:
        print(f"Error: Could not connect to the MongoDB server. Details: {e}")
    except ConfigurationError as e:
        print(f"Error: Invalid configuration or MongoDB URI. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        
    # Return None if the connection fails, allowing the calling code to handle the failure gracefully
    return None


def create_mongodb_vector_db(json_filepath, collection):
    print("1. Loading embedding model...")
    model = TextEmbedding(model_name=embedding_model)
    
    print("2. Parsing JSON schema into text documents...")
    documents = parse_schema_to_documents(json_filepath)
    
    print(f"3. Generating embeddings for {len(documents)} tables...")
    for idx, doc in enumerate(documents):
        # Explicitly ensure the embedding is a plain Python list of floats
        embedding_vector = next(model.embed(doc["content"])).tolist()
        doc["embedding"] = embedding_vector
    
    print("4. Clearing existing data...")
    collection.delete_many({})
    
    print("5. Inserting documents into MongoDB...")
    try:
        # Pass a completely clean, native list of dictionaries
        collection.insert_many(documents)
        print(f"Successfully inserted {len(documents)} tables!")
    except Exception as e:
        print(f"Insertion failed! Error details: {e}")


def create_vector_index(collection: Collection, index_name: str, dimensions: int):
    
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": "cosine"       # Similarity metric: 'cosine', 'euclidean', or 'dotProduct'
            }
        ]
    }

    # Create the Search Index Model
    # We set type="vectorSearch" explicitly for Atlas Vector Search
    search_index_model = SearchIndexModel(
        definition=index_definition,
        name = index_name,
        type="vectorSearch"
    )

    try:
        print(f"Initiating vector search index '{index_name}' on '{collection}'...")
        
        check_index = list(collection.list_search_indexes())
        # 3. Request index creation from MongoDB Atlas
        if len(check_index) == 0:
            result_name = collection.create_search_index(model=search_index_model)
        else:
            return check_index[0].get('name')
        print(f"Index creation request acknowledged by Atlas. Index: {result_name}")
        # 4. Polling loop to wait until the index status transitions to 'READY'
        print("Waiting for index build to complete (this may take a few minutes)...")
        while True:
            # Query Atlas for the status of this specific index
            indices = list(collection.list_search_indexes(name=index_name))
            
            if indices:
                status = indices[0].get("status")
                print(f"Current compilation status: {status}")
                
                if status == "READY":
                    print(f"SUCCESS! Vector search index '{index_name}' is fully active.")
                    return result_name
                elif status == "FAILED":
                    # Extract the error reason provided by Atlas if available
                    fail_reason = indices[0].get("queryable_status_desc", "Unknown error")
                    raise RuntimeError(f"Atlas Vector Search index build failed: {fail_reason}")
            
            # Wait 10 seconds before checking the status again to prevent spamming the cluster
            time.sleep(10)
    except PyMongoError as pm_err:
        print(f"General PyMongo Driver Error occurred while building the index.")
        raise pm_err
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e