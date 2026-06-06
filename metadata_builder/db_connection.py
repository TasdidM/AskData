import os
import logging
from dotenv import load_dotenv

# Initialize a logger for this module
# (In a larger app, you might configure this centrally)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url() -> str:
    """
    Loads environment variables and constructs the database connection string.
    
    Returns:
        str: The formatted SQLAlchemy connection URL.
    """
    # Load variables from the .env file into the environment
    load_dotenv()

    # Retrieve database configuration with safe fallbacks
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "my_database")

    # Construct the actual connection string used by the application
    connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # DEBUGGING: Construct a safe version of the URL with the password masked
    # This ensures other developers can see where the app is trying to connect 
    # without leaking sensitive credentials into the console or log files.
    masked_url = f"mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}"
    logger.info(f"Initializing database connection: {masked_url}")

    return connection_url

if __name__ == "__main__":
    # Example usage
    URL = get_database_url()