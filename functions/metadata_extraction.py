import json
from pathlib import Path
from sqlalchemy import create_engine, inspect, MetaData, Table, select, func

from typing import Any, Dict, List, Optional
from sqlalchemy.engine import Connection, Inspector


def _serialize_value(val: Any) -> Any:
    """Converts non-standard database types (e.g., Decimal, DateTime, UUID) 

    into primitive strings or returns native primitives directly.
    """
    if isinstance(val, (int, float, bool, str)) or val is None:
        return val
    return str(val)


def _fetch_sample_rows(connection: Connection, table: Table, sample_size: int) -> List[Dict[str, Any]]:
    """Retrieves a random sample of rows from a specific table."""
    try:
        # Note: func.random() might trigger full table scans on very large tables
        query = select(table).order_by(func.random()).limit(sample_size)
        result = connection.execute(query)
        
        return [
            {key: _serialize_value(val) for key, val in row.items()}
            for row in result.mappings()
        ]
    except Exception as err:
        print(f"  ⚠️ Warning: Could not fetch sample data for '{table.name}': {err}")
        return []


def _fetch_low_cardinality_values(
    connection: Connection, table: Table, col_name: str, max_cardinality: int = 30
) -> Optional[List[Any]]:
    """Returns unique column values if the column's total cardinality is 

    strictly lower than the specified threshold. Otherwise, returns None.
    """
    try:
        # Check up to max_cardinality limit to evaluate threshold safely
        query = select(table.c[col_name]).distinct().limit(max_cardinality)
        distinct_results = connection.execute(query).scalars().all()
        
        if len(distinct_results) < max_cardinality:
            return [_serialize_value(v) for v in distinct_results]
            
    except Exception:
        # Silently bypass un-queriable data columns (e.g., BLOBs, raw JSON fields)
        pass
    return None


def _profile_table_columns(
    connection: Connection, table_name: str, raw_columns: List[Dict[str, Any]], metadata_obj: MetaData
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reflects the database table structure to analyze data distributions 

    and extract unique categoricals alongside row samples.
    """
    columns_metadata = []
    sample_rows = []
    
    try:
        table = Table(table_name, metadata_obj, autoload_with=connection)
        sample_rows = _fetch_sample_rows(connection, table, len(raw_columns))
        
        for col in raw_columns:
            col_name = col['name']
            col_dict = {
                "name": col_name,
                "type": str(col['type']),
                "nullable": col['nullable']
            }
            
            unique_vals = _fetch_low_cardinality_values(connection, table, col_name)
            if unique_vals is not None:
                col_dict["unique_values"] = unique_vals
                
            columns_metadata.append(col_dict)
            
    except Exception as table_err:
        print(f"  ⚠️ Warning: Table reflection skipped or failed for '{table_name}': {table_err}")
        # Structural fallback if data inspection fails
        columns_metadata = [
            {"name": c['name'], "type": str(c['type']), "nullable": c['nullable']} 
            for c in raw_columns
        ]
        
    return columns_metadata, sample_rows


def _extract_table_metadata(
    inspector: Inspector, connection: Connection, metadata_obj: MetaData, table_name: str, sample_size: int
) -> Dict[str, Any]:
    """Orchestrates structural extraction and data profiling for an individual table."""
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
    """Connects to a target database to inspect schemas, map internal relational 

    dependencies, and generate descriptive data profiles for LLM context.
    """
    try:
        engine = create_engine(connection_url)
        metadata_obj = MetaData()
        db_metadata = {}
        
        with engine.connect() as connection:
            inspector = inspect(connection)
            
            for table_name in inspector.get_table_names():
                print(f"Extracting schema & data for: {table_name}")
                db_metadata[table_name] = _extract_table_metadata(
                    inspector, connection, metadata_obj, table_name, sample_size
                )
                
        return db_metadata
        
    except Exception as e:
        print(f"🔴 Critical: Root architecture extraction failure: {e}")
        return {}


def generate_schema_from_db(db_url: str) -> Dict:
    # 1. Connect to your database
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    schema_dict = {}
    
    # 2. Iterate through all tables
    for table_name in inspector.get_table_names():
        schema_dict[table_name] = {}
        
        # 3. Fetch columns for each table
        for column in inspector.get_columns(table_name):
            col_name = column['name']
            # Convert SQLAlchemy type object to a string representation (e.g., VARCHAR, INTEGER)
            col_type = str(column['type']).split('(')[0].upper() 
            
            schema_dict[table_name][col_name] = col_type
            
    return schema_dict

def export_schema_to_json(schema: Dict, file_path: str = "db_base_schema.json", indent: int = 2, overwrite: bool = True) -> Path:
    """
    Saves a schema dictionary to a JSON file.

    Args:
        schema: The schema dictionary to save.
        file_path: Destination file path.
        indent: JSON indentation level.
        overwrite: If False and the file exists, a FileExistsError is raised.

    Returns:
        Path object pointing to the written file.
    """
    path = Path(file_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    # Ensure parent directory exists
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=indent, ensure_ascii=False)

    return path


# def extract_db_metadata(connection_url: str, sample_size: int = 30) -> dict:
    """
    Si collega a un database, estrae la struttura, un campione di righe,
    e i valori univoci per le colonne con cardinalità < 30 (es. categorie, status).
    """
    try:
        engine = create_engine(connection_url)
        metadata_obj = MetaData()
        db_metadata = {}
        
        with engine.connect() as connection:
            inspector = inspect(connection)
            
            for table_name in inspector.get_table_names():
                print(f"Extracting schema & data for: {table_name}")
                
                # --- 1. Extract Base Metadata ---
                raw_columns = inspector.get_columns(table_name)
                fks = inspector.get_foreign_keys(table_name)
                pk = inspector.get_pk_constraint(table_name)
                
                sample_rows = []
                columns_metadata = []
                
                # --- 2. Reflect Table to query Data ---
                try:
                    table = Table(table_name, metadata_obj, autoload_with=connection)
                    
                    # 2a. Grab Random Sample Rows
                    try:
                        query = select(table).order_by(func.random()).limit(sample_size)
                        result = connection.execute(query)
                        
                        for row in result.mappings():
                            clean_row = {}
                            for key, val in row.items():
                                if isinstance(val, (int, float, bool, str)) or val is None:
                                    clean_row[key] = val
                                else:
                                    clean_row[key] = str(val)
                            sample_rows.append(clean_row)
                    except Exception as err:
                        print(f"  ⚠️ Warning: Could not fetch sample data: {err}")

                    # 2b. Extract Unique Values for Columns (< 30)
                    for col in raw_columns:
                        col_name = col['name']
                        col_dict = {
                            "name": col_name, 
                            "type": str(col['type']), 
                            "nullable": col['nullable']
                        }
                        
                        try:
                            # The LIMIT 30 is crucial. It stops the database from doing 
                            # a massive table scan once it hits 30 unique items.
                            dist_query = select(table.c[col_name]).distinct().limit(30)
                            dist_result = connection.execute(dist_query).scalars().all()
                            
                            # If it found less than 30, we have the complete list of unique values!
                            if len(dist_result) < 30:
                                unique_vals = [
                                    v if isinstance(v, (int, float, bool, str)) or v is None else str(v) 
                                    for v in dist_result
                                ]
                                col_dict["unique_values"] = unique_vals
                        except Exception as err:
                            # Silently skip columns that can't be distinct-queried (like JSON/BLOBs)
                            pass
                            
                        columns_metadata.append(col_dict)

                except Exception as table_err:
                    print(f"  ⚠️ Warning: Could not reflect table {table_name}: {table_err}")
                    # If table reflection fails, fallback to just structural metadata without unique values
                    columns_metadata = [
                        {
                            "name": c['name'], "type": str(c['type']), "nullable": c['nullable']
                        } for c in raw_columns
                    ]

                # --- 3. Bundle the Table Data ---
                db_metadata[table_name] = {
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
                
        return db_metadata
        
    except Exception as e:
        print(f"Errore generale durante l'estrazione: {e}")
        return {}