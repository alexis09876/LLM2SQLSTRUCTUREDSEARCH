import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dataclasses import dataclass
from typing import Optional

@dataclass
class EdgeRecord:
    db_id: str
    child_table: str
    child_column: str
    child_col_id: int
    parent_table: str
    parent_column: str
    parent_col_id: int
    kind: str = "explicit_fk"
    confidence: float = 1.0
    notes: Optional[str] = "from dev_tables.json"