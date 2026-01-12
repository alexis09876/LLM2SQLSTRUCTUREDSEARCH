import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Any, Dict, List, Set, Tuple

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.embedding.embedding_client import EmbeddingClient


class SQLStructureEmbeddingBuilder:
    """
    Build K:V embeddings for tables and columns from an LLM JSON output that contains
    SQL candidates with per-candidate tables/columns/column_relationships.

    Output JSON schema (columns grouped by table, column embeddings are column-only):
    {
      "intent": "<...>",
      "tables": { "<table_name>": { "embedding": [...] }, ... },
      "columns": {
        "<table_name>": {
          "<column_name>": { "embedding": [...] },
          ...
        },
        ...
      },
      "relationships": [
        {
          "left":  { "table": "<t1>", "column": "<c1>" },
          "right": { "table": "<t2>", "column": "<c2>" },
          "relation": "<...>",
          "context": "<...>",
          "note": "<...>"
        }
      ],
      "source_notes": "<...>"
    }
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """
        Initialize the builder with an embedding client.

        Args:
            model_name (str): The embedding model name. Defaults to 'text-embedding-3-small'.
        """
        self.embedder = EmbeddingClient(model_name=model_name)

    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Read one input JSON file, extract unique table names, extract columns grouped by table
        (columns must be qualified as '<table>.<column>'), normalize column↔column relationships,
        compute embeddings, and write a compact grouped K:V JSON to disk.

        Expected input JSON (per file):
        {
          "intent": "...",
          "candidates": [
            {
              "sql": "...",
              "tables": ["employees", "departments", ...],
              "columns": ["employees.employee_id", "departments.department_id", ...],
              "column_relationships": [
                { "left": "employees.department_id", "right": "departments.department_id",
                  "relation": "equals", "context": "join", "note": "..." }
              ]
            }
          ],
          "notes": "..."
        }

        Args:
            input_path (str): Path to the input JSON produced by the LLM.
            output_path (str): Path to save the new JSON with grouped K:V embeddings.

        Returns:
            Dict[str, Any]: The normalized JSON object that was written to disk.

        Raises:
            ValueError: If the input JSON is invalid or missing required fields.
            RuntimeError: If the embedding count does not match the number of names.
        """
        data = FileReader.load_json(input_path)
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise ValueError("Input JSON must be an object with a 'candidates' list.")

        intent = str(data.get("intent", "") or "")
        notes = str(data.get("notes", "") or "")

        # Collect unique tables and columns grouped by table
        table_set: Set[str] = set()
        table_to_cols: Dict[str, Set[str]] = {}
        relationships: List[Dict[str, Any]] = []

        for cand in data["candidates"]:
            if not isinstance(cand, dict):
                continue

            # Tables
            for t in (cand.get("tables") or []):
                if isinstance(t, str):
                    tname = t.strip()
                    if tname:
                        table_set.add(tname)

            # Columns (require qualified form '<table>.<column>')
            for fq in (cand.get("columns") or []):
                t, c = self._split_qualified_name(fq)
                if not t or not c:
                    # Skip unqualified/invalid columns per your assumption
                    continue
                if t not in table_to_cols:
                    table_to_cols[t] = set()
                table_to_cols[t].add(c)

            # Relationships
            for rel in (cand.get("column_relationships") or []):
                norm = self._normalize_relationship(rel)
                if norm:
                    relationships.append(norm)

        # Embed tables
        tables_map = self._embed_names(sorted(table_set))

        # Embed columns strictly by column name (column-only, no table mixed in)
        unique_column_names: List[str] = sorted({c for cols in table_to_cols.values() for c in cols})
        column_vectors = self._embed_names(unique_column_names)  # keys are column names

        # Build grouped columns map using column-only vectors
        grouped_columns_map: Dict[str, Dict[str, Any]] = {}
        for t in sorted(table_to_cols.keys()):
            grouped_columns_map[t] = {}
            for c in sorted(table_to_cols[t]):
                grouped_columns_map[t][c] = {
                    "embedding": column_vectors.get(c, {}).get("embedding", [])
                }

        out_obj: Dict[str, Any] = {
            "intent": intent,
            "tables": tables_map,
            "columns": grouped_columns_map,
            "relationships": relationships,
            "source_notes": notes
        }

        FileWriter.write_json(out_obj, output_path)
        return out_obj

    # ------------------------- helpers (used) -------------------------

    @staticmethod
    def _extract_column_name(fq_col: Any) -> str:
        """
        Extract the column name from a string like 'table.column'.
        If unqualified, return the string itself; return '' if invalid.

        Args:
            fq_col (Any): Fully- or un-qualified column string.

        Returns:
            str: Column-only name (may be empty if invalid).
        """
        if not isinstance(fq_col, str):
            return ""
        text = fq_col.strip()
        if not text:
            return ""
        if "." not in text:
            return text
        return text.split(".", 1)[1].strip()

    @staticmethod
    def _split_qualified_name(name: Any) -> Tuple[str, str]:
        """
        Split '<table>.<column>' into (table, column). If unqualified, ('', name).

        Args:
            name (Any): Qualified or unqualified name.

        Returns:
            Tuple[str, str]: (table, column)
        """
        if not isinstance(name, str):
            return "", ""
        text = name.strip()
        if not text:
            return "", ""
        if "." not in text:
            return "", text
        t, c = text.split(".", 1)
        return t.strip(), c.strip()

    def _normalize_relationship(self, rel: Any) -> Dict[str, Any]:
        """
        Normalize a relationship dict that contains 'left' and 'right' into a structured form.

        Args:
            rel (Any): Original relationship object.

        Returns:
            Dict[str, Any]: {
                "left":  {"table": "<t1>", "column": "<c1>"},
                "right": {"table": "<t2>", "column": "<c2>"},
                "relation": "<...>",
                "context": "<...>",
                "note": "<...>"
            }
            or {} if invalid.
        """
        if not isinstance(rel, dict):
            return {}
        lt, lc = self._split_qualified_name(rel.get("left", ""))
        rt, rc = self._split_qualified_name(rel.get("right", ""))
        return {
            "left": {"table": lt, "column": lc},
            "right": {"table": rt, "column": rc},
            "relation": rel.get("relation", ""),
            "context": rel.get("context", ""),
            "note": rel.get("note", "")
        }

    def _embed_names(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compute embeddings for a list of unique names and return a K:V map:
        { "<name>": {"embedding": [...]}, ... }

        Args:
            names (List[str]): Unique names.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from name to {"embedding": vector}.

        Raises:
            RuntimeError: If the returned vector count mismatches the input count.
        """
        if not names:
            return {}
        vectors = self.embedder.embed(names)
        if len(vectors) != len(names):
            raise RuntimeError(f"Embedding count mismatch: expected {len(names)}, got {len(vectors)}.")
        return {name: {"embedding": vec} for name, vec in zip(names, vectors)}


if __name__ == "__main__":
    builder = SQLStructureEmbeddingBuilder(model_name="text-embedding-3-small")

    input_file = "backend/src/query/output/sql_candidates.json"
    output_file = "backend/src/query/output/sql_structure_embeddings.json"

    try:
        out = builder.process(input_file, output_file)
        print("[INFO] Embedding process completed.")
        print("[INFO] Output saved to:", output_file)

        relationships = out.get("relationships", [])
        if relationships:
            rel = relationships[0]
            left_table = rel["left"]["table"]
            left_col = rel["left"]["column"]
            right_table = rel["right"]["table"]
            right_col = rel["right"]["column"]

            left_table_emb = out["tables"].get(left_table, {}).get("embedding", [])
            left_col_emb = (
                out["columns"].get(left_table, {})
                              .get(left_col, {})
                              .get("embedding", [])
            )
            right_table_emb = out["tables"].get(right_table, {}).get("embedding", [])
            right_col_emb = (
                out["columns"].get(right_table, {})
                              .get(right_col, {})
                              .get("embedding", [])
            )

            relation_embedding = left_table_emb + left_col_emb + right_table_emb + right_col_emb

            print("[DEBUG] First relationship:", rel)
            print("[DEBUG] Relation embedding length:", len(relation_embedding))
        else:
            print("[INFO] No relationships found in this file.")

    except Exception as e:
        print("❌ Error during processing:", str(e))
