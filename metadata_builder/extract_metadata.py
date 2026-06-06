import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, inspect, MetaData, Table, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
# Setting up a logger makes debugging much easier. You can change 
# logging.INFO to logging.DEBUG when you need to inspect deeper execution flows.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def _serialize_value(val: Any) -> Any:
    """
    Converts non-standard database types (e.g., Decimal, DateTime, UUID) 
    into primitive strings or returns native primitives directly.
    
    This ensures the final output is JSON-serializable.
    """
    if isinstance(val, (int, float, bool, str)) or val is None:
        return val
    return str(val)


def _fetch_sample_rows(connection: Connection, table: Table, sample_size: int) -> List[Dict[str, Any]]:
    """
    Retrieves a random sample of rows from a specific table.

    Args:
        connection: Active SQLAlchemy connection.
        table: The SQLAlchemy Table object to query.
        sample_size: Number of random rows to fetch.

    Returns:
        A list of dictionaries representing the sampled rows.
    """
    try:
        # Note for optimization: func.random() can trigger full table scans.
        # If querying massive tables (>1M rows) becomes a bottleneck, 
        # consider replacing this with a direct LIMIT query or TABLESAMPLE if supported.
        logger.debug(f"Fetching {sample_size} sample rows for table '{table.name}'...")
        query = select(table).order_by(func.random()).limit(sample_size)
        result = connection.execute(query)
        
        # Map over the rows and serialize complex types (like Datetime) to strings
        return [
            {key: _serialize_value(val) for key, val in row.items()}
            for row in result.mappings()
        ]
    except Exception as err:
        # Log the error with exc_info=True to capture the stack trace for debugging
        logger.warning(f"Could not fetch sample data for '{table.name}': {err}", exc_info=True)
        return []


def _fetch_low_cardinality_values(
    connection: Connection, table: Table, col_name: str, max_cardinality: int = 30
) -> Optional[List[Any]]:
    """
    Returns unique column values if the column's total cardinality is 
    strictly lower than the specified threshold. Useful for finding ENUMs or categories.

    Args:
        connection: Active SQLAlchemy connection.
        table: The SQLAlchemy Table object.
        col_name: Name of the column to profile.
        max_cardinality: The maximum number of unique values allowed before bailing out.

    Returns:
        List of distinct values if under threshold, otherwise None.
    """
    try:
        # We limit the query to (max_cardinality) to avoid pulling millions of 
        # distinct records into memory unnecessarily.
        query = select(table.c[col_name]).distinct().limit(max_cardinality)
        distinct_results = connection.execute(query).scalars().all()
        
        # If the number of distinct results hit the limit, it's high-cardinality.
        if len(distinct_results) < max_cardinality:
            return [_serialize_value(v) for v in distinct_results]
            
    except Exception as err:
        # Silently bypass un-queriable data columns (e.g., BLOBs, raw JSON fields)
        # We use debug here instead of warning so we don't clutter logs on expected failures.
        logger.debug(f"Skipping distinct value check for {table.name}.{col_name}: {err}")
        
    return None


# ---------------------------------------------------------
# Core Extraction Logic
# ---------------------------------------------------------

def _profile_table_columns(
    connection: Connection, table_name: str, raw_columns: List[Dict[str, Any]], metadata_obj: MetaData
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reflects the database table structure to analyze data distributions 
    and extract unique categoricals alongside row samples.
    """
    columns_metadata = []
    sample_rows = []
    
    try:
        # Reflect the table structure from the database
        table = Table(table_name, metadata_obj, autoload_with=connection)
        
        # Extract a dynamic sample size based on column count (or customize this logic)
        sample_rows = _fetch_sample_rows(connection, table, len(raw_columns))
        
        # Analyze each column
        for col in raw_columns:
            col_name = col['name']
            col_dict = {
                "name": col_name,
                "type": str(col['type']),
                "nullable": col['nullable']
            }
            
            # Check for categories/enums if cardinality is low enough
            unique_vals = _fetch_low_cardinality_values(connection, table, col_name)
            if unique_vals is not None:
                col_dict["unique_values"] = unique_vals
                
            columns_metadata.append(col_dict)
            
    except Exception as table_err:
        logger.warning(f"Table reflection skipped or failed for '{table_name}': {table_err}")
        # Structural fallback: Provide basic metadata even if data inspection (profiling) fails
        columns_metadata = [
            {"name": c['name'], "type": str(c['type']), "nullable": c['nullable']} 
            for c in raw_columns
        ]
        
    return columns_metadata, sample_rows


def _extract_table_metadata(
    inspector: Inspector, connection: Connection, metadata_obj: MetaData, table_name: str, sample_size: int
) -> Dict[str, Any]:
    """
    Orchestrates structural extraction and data profiling for an individual table.
    Collects columns, Primary Keys (PKs), Foreign Keys (FKs), and sample data.
    """
    raw_columns = inspector.get_columns(table_name)
    fks = inspector.get_foreign_keys(table_name)
    pk = inspector.get_pk_constraint(table_name)
    
    columns_metadata, sample_rows = _profile_table_columns(
        connection, table_name, raw_columns, metadata_obj
    )
    
    return {
        "columns": columns_metadata,
        "primary_key": pk.get("constrained_columns", []),
        "relationships": [
            {
                "from_column": fk['constrained_columns'],
                "to_table": fk['referred_table'],
                "to_column": fk['referred_columns']
            } for fk in fks
        ],
        "sample_data": sample_rows
    }


def extract_db_metadata(connection_url: str, sample_size: int = 30) -> Dict[str, Any]:
    """
    Connects to a target database to inspect schemas, map internal relational 
    dependencies, and generate descriptive data profiles for LLM context.
    """
    try:
        engine = create_engine(connection_url)
        metadata_obj = MetaData()
        db_metadata = {}
        
        with engine.connect() as connection:
            inspector = inspect(connection)
            
            for table_name in inspector.get_table_names():
                logger.info(f"Extracting detailed schema & data for: {table_name}")
                db_metadata[table_name] = _extract_table_metadata(
                    inspector, connection, metadata_obj, table_name, sample_size
                )
                
        return db_metadata
        
    except Exception as e:
        # Use logger.error to explicitly highlight connection or critical failures
        logger.error(f"Critical: Root architecture extraction failure.", exc_info=True)
        return {}


def generate_schema_from_db(db_url: str) -> Dict[str, Any]:
    """
    Generates a simplified, flattened dictionary mapping of the database schema 
    (tables and their column types), specifically formatted for tools like SQLGlot.
    """
    logger.info("Generating flat schema map...")
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    schema_dict = {}
    
    for table_name in inspector.get_table_names():
        # Lowercase the table name to ensure compatibility with parsing tools (e.g. SQLGlot)
        normalized_table = table_name.lower()
        schema_dict[normalized_table] = {}
        
        for column in inspector.get_columns(table_name):
            col_name = column['name'].lower()
            
            # Convert SQLAlchemy type object to a simplified string representation 
            # E.g., VARCHAR(50) -> VARCHAR
            col_type = str(column['type']).split('(')[0].upper() 
            schema_dict[normalized_table][col_name] = col_type
            
    return schema_dict


# ---------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------

def export_schema_to_json(
    schema: Dict[str, Any], file_path: str = "assets/db_base_schema.json", 
    indent: int = 2, overwrite: bool = True
) -> Path:
    """
    Saves a schema dictionary to a JSON file.

    Args:
        schema: The schema dictionary to save.
        file_path: Destination file path.
        indent: JSON indentation level for readability.
        overwrite: If False and the file exists, a FileExistsError is raised.

    Returns:
        Path object pointing to the written file.
    """
    path = Path(file_path)
    
    if path.exists() and not overwrite:
        logger.error(f"Cannot export: File already exists at {path}")
        raise FileExistsError(f"File already exists: {path}")

    # Ensure parent directory tree exists before attempting to write
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=indent, ensure_ascii=False)
        logger.info(f"Successfully exported schema to {path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write JSON file to {path}: {e}", exc_info=True)
        raise

    return path


# ---------------------------------------------------------
# Example Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # DB_URL = "sqlite:///example.db"
    # metadata = extract_db_metadata(DB_URL)
    # export_schema_to_json(metadata, "assets/extracted_metadata.json")
    pass