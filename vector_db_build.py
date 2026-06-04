from dotenv import load_dotenv
from functions.create_vectordb import connect_mongodb_collection, create_mongodb_vector_db, create_vector_index
import os
load_dotenv()

db_schema_json_path = os.getenv("DB_SCHEMA_JSON_PATH")
mongo_uri = os.getenv("MONGO_URI")
vector_db_name = os.getenv("VECTOR_DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
vector_index_name = os.getenv("VECTOR_INDEX_NAME")

mongodb_collection = connect_mongodb_collection(mongo_uri, vector_db_name, collection_name)


# Transform detailed schema in json to documents of str for embeddings and upload in mongodb
#create_mongodb_vector_db(db_schema_json_path, mongodb_collection)

# create the vector search indexing for the documents
# create_vector_index(mongodb_collection, vector_index_name, 384)

