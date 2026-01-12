import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
from typing import Any, Dict, List

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.embedding.embedding_client import EmbeddingClient


class ColumnNamesEmbedder:
    """
    Generate name-level embeddings for column records.

    Input:  columns.json  (full list from DevSchemaCatalog)
    Output: columns_name_embeddings.json
    """

    def __init__(self, embedder: EmbeddingClient) -> None:
        """
        Initialize with an EmbeddingClient.

        Args:
            embedder (EmbeddingClient): The embedding client wrapper.
        """
        self.embedder = embedder

    @staticmethod
    def _build_col_key(db_id: str, table: str, column: str) -> str:
        """Build a stable key for a column: db.table.column."""
        return f"{db_id}.{table}.{column}"

    def run(self, columns_json_path: str, out_path: str) -> None:
        """
        Generate and write name-level embeddings for columns.

        Args:
            columns_json_path (str): Path to columns.json.
            out_path (str): Path to output file.

        Returns:
            None
        """
        rows: List[Dict[str, Any]] = FileReader.load_json(columns_json_path)
        if not rows:
            FileWriter.write_json([], out_path)
            return

        uniq_db = sorted({r["db_id"] for r in rows})
        uniq_table = sorted({r["table"] for r in rows})
        uniq_col = sorted({r["column"] for r in rows})

        db_vecs = self.embedder.embed(uniq_db)
        tbl_vecs = self.embedder.embed(uniq_table)
        col_vecs = self.embedder.embed(uniq_col)

        db_map = {n: v for n, v in zip(uniq_db, db_vecs)}
        tbl_map = {n: v for n, v in zip(uniq_table, tbl_vecs)}
        col_map = {n: v for n, v in zip(uniq_col, col_vecs)}

        dim = len(db_vecs[0]) if db_vecs else 0
        ts = int(time.time())

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "db_id": r["db_id"],
                "table": r["table"],
                "column": r["column"],
                "col_id": r.get("col_id"),
                "key": self._build_col_key(r["db_id"], r["table"], r["column"]),

                "db_id_vec": db_map[r["db_id"]],
                "table_vec": tbl_map[r["table"]],
                "column_vec": col_map[r["column"]],

                "model": self.embedder.model_name,
                "dim": dim,
                "created_at": ts
            })

        FileWriter.write_json(out, out_path)


if __name__ == "__main__":
    columns_json = "backend/src/schema_parser/output/columns.json"
    out_path = "backend/src/embedding/output/columns_name_embeddings.json"

    embedder = EmbeddingClient(model_name="text-embedding-3-small")
    ColumnNamesEmbedder(embedder).run(columns_json, out_path)
    print("[OK] wrote", out_path)
