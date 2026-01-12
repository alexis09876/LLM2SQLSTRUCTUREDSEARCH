import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
from typing import Any, Dict, List

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.embedding.embedding_client import EmbeddingClient


class EdgeNamesEmbedder:
    """
    Generate name-level embeddings for edge (FK) records.

    Input:  edges.json
    Output: edges_name_embeddings.json
    """

    def __init__(self, embedder: EmbeddingClient) -> None:
        self.embedder = embedder

    @staticmethod
    def _build_edge_key(db_id: str, child_table: str, child_column: str,
                        parent_table: str, parent_column: str) -> str:
        """Build stable key: db:child_table.child_column->parent_table.parent_column."""
        return f"{db_id}:{child_table}.{child_column}->{parent_table}.{parent_column}"

    def run(self, edges_json_path: str, out_path: str) -> None:
        """
        Generate and write name-level embeddings for edges.

        Args:
            edges_json_path (str): Path to edges.json.
            out_path (str): Path to output file.

        Returns:
            None
        """
        rows: List[Dict[str, Any]] = FileReader.load_json(edges_json_path)
        if not rows:
            FileWriter.write_json([], out_path)
            return

        uniq_db = sorted({r["db_id"] for r in rows})
        uniq_tbl = sorted({t for r in rows for t in (r["child_table"], r["parent_table"])})
        uniq_col = sorted({c for r in rows for c in (r["child_column"], r["parent_column"])})

        db_vecs = self.embedder.embed(uniq_db)
        tbl_vecs = self.embedder.embed(uniq_tbl)
        col_vecs = self.embedder.embed(uniq_col)

        db_map = {n: v for n, v in zip(uniq_db, db_vecs)}
        tbl_map = {n: v for n, v in zip(uniq_tbl, tbl_vecs)}
        col_map = {n: v for n, v in zip(uniq_col, col_vecs)}

        dim = len(db_vecs[0]) if db_vecs else 0
        ts = int(time.time())

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "db_id": r["db_id"],
                "child_table": r["child_table"],
                "child_column": r["child_column"],
                "parent_table": r["parent_table"],
                "parent_column": r["parent_column"],
                "edge_key": self._build_edge_key(r["db_id"], r["child_table"],
                                                 r["child_column"], r["parent_table"], r["parent_column"]),

                "db_id_vec": db_map[r["db_id"]],
                "child_table_vec": tbl_map[r["child_table"]],
                "child_column_vec": col_map[r["child_column"]],
                "parent_table_vec": tbl_map[r["parent_table"]],
                "parent_column_vec": col_map[r["parent_column"]],

                "model": self.embedder.model_name,
                "dim": dim,
                "created_at": ts
            })

        FileWriter.write_json(out, out_path)


if __name__ == "__main__":
    edges_json = "backend/src/schema_parser/output/edges.json"
    out_path = "backend/src/embedding/output/edges_name_embeddings.json"

    embedder = EmbeddingClient(model_name="text-embedding-3-small")
    EdgeNamesEmbedder(embedder).run(edges_json, out_path)
    print("[OK] wrote", out_path)
