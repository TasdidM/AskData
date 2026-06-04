from dotenv import load_dotenv
import os

# carica i varibile presente nel .env 
load_dotenv()


# Configutazione – sovrascrivere tramite variabili .env o modificare direttamente
DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "3306")
DB_NAME     = os.getenv("DB_NAME",     "my_database")

# crea la stringa di connessione con database
CONNECTION_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)