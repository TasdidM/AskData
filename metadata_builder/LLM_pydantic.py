from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# =====================================================================
# SECTION 1: Database Structural Metadata Models
# Used to map the physical structure and relationships of the database.
# =====================================================================

class ColumnMetadata(BaseModel):
    """Metadata defining the structure and AI-enriched details of a single database column."""
    name: str = Field(..., description="The physical name of the column in the database.")
    type: str = Field(..., description="The SQL data type of the column (e.g., VARCHAR, INT).")
    nullable: bool = Field(..., description="Indicates if the column allows NULL values (True) or not (False).")
    
    # AI-Enriched Fields
    description: Optional[str] = Field(None, description="LLM-generated description of the column's business logic.")
    synonyms: Optional[str] = Field(None, description="LLM-generated synonyms of the column name.")

class RelationshipMetadata(BaseModel):
    """Defines a foreign key relationship linking one table to another."""
    from_column: List[str] = Field(..., description="List of columns in the current table that act as the foreign key.")
    to_table: str = Field(..., description="The name of the target table this relationship points to.")
    to_column: List[str] = Field(..., description="List of referenced columns in the target table.")

class TableMetadata(BaseModel):
    """Metadata defining a single database table, including its columns and relationships."""
    name: str = Field(..., description="The physical name of the table in the database.")
    columns: List[ColumnMetadata] = Field(..., description="List of metadata definitions for the columns in this table.")
    primary_key: List[str] = Field(default_factory=list, description="List of column names that make up the primary key.")
    relationships: List[RelationshipMetadata] = Field(default_factory=list, description="List of outward foreign key relationships.")
    
    # AI-Enriched Fields
    description: Optional[str] = Field(None, description="LLM-generated description of the table's purpose.")
    synonyms: Optional[str] = Field(None, description="LLM-generated synonyms of the table name.")

class DatabaseSchema(BaseModel):
    """Root model representing the entire schema of a database."""
    tables: Dict[str, TableMetadata] = Field(..., description="Dictionary mapping table names to their structural metadata.")


# =====================================================================
# SECTION 2: LLM Output Models
# Used strictly to enforce the schema for LLM text generation tasks.
# =====================================================================

class ColumnDescription(BaseModel):
    """Payload model representing the LLM's generated context for a specific column."""
    name: str = Field(..., description="The name of the column as provided in the input prompt.")
    description: str = Field(..., description="A detailed, verbose explanation of what this column stores and its business logic.")
    synonyms: Optional[str] = Field(None, description="LLM-generated synonyms for the column.")

class TableDescription(BaseModel):
    """Payload model representing the LLM's generated context for an entire table."""
    name: str = Field(..., description="The name of the table as provided in the input prompt.")
    description: str = Field(..., description="A high-level verbose overview of the table's purpose and contents.")
    synonyms: Optional[str] = Field(None, description="LLM-generated synonyms for the table.")
    columns: List[ColumnDescription] = Field(..., description="List of generated descriptions for the table's columns.")