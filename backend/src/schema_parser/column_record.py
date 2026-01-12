import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class ColumnRecord:
    db_id: str
    table: str
    column: str
    col_id: int
    ordinal_in_table: Optional[int]
    type: str

    # keys
    is_pk: bool
    pk_group_id: Optional[str]
    is_fk: bool
    fk_ref_table: Optional[str]
    fk_ref_column: Optional[str]
    fk_ref_col_id: Optional[int]
    fk_confidence: Optional[float]  # 1.0 for explicit FK, None otherwise

    # placeholders for later enrichment
    table_desc: Optional[str] = None
    column_desc: Optional[str] = None
    aliases: Optional[List[str]] = None

    value_type: Optional[str] = None
    cardinality: Optional[int] = None
    uniqueness_ratio: Optional[float] = None
    null_rate: Optional[float] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    enum_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    unit: Optional[str] = None
    example_values: Optional[List[Any]] = None

    sql_hints: Optional[List[str]] = None
    notes: Optional[str] = None