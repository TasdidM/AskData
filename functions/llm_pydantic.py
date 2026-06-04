from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ColumnMetadata(BaseModel):
    name: str
    type: str
    nullable: bool
    description: Optional[str] = Field(None, description="LLM generated description of the column")
    synonyms: Optional[str] = Field(None, description = "LLM generated sysnonyms of the column")

class RelationshipMetadata(BaseModel):
    from_column: List[str]
    to_table: str
    to_column: List[str]

class TableMetadata(BaseModel):
    name: str
    description: Optional[str] = Field(None, description="LLM generated description of the table")
    synonyms: Optional[str] = Field(None, description = "LLM generated sysnonyms of the table")
    columns: List[ColumnMetadata]
    primary_key: List[str]
    relationships: List[RelationshipMetadata]

class DatabaseSchema(BaseModel):
    tables: Dict[str, TableMetadata]




class ColumnDescription(BaseModel):
    name: str = Field(..., description="The name of the column as provided in the input")
    description: str = Field(..., description="A detailed, verbose explanation of what this column stores and its business logic")
    synonyms: Optional[str] = Field(..., description = "LLM generated sysnonyms of the column")

class TableDescription(BaseModel):
    name: str = Field(..., description="The name of the table")
    description: str = Field(..., description="A high-level verbose overview of the table's purpose and contents")
    synonyms: Optional[str] = Field(..., description = "LLM generated sysnonyms of the table")
    columns: List[ColumnDescription]
    