import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set

from backend.src.data_io.file_reader import FileReader
from backend.src.utils.utils import Utils


class CandidateGraphBuilder:
    """
    Build candidate tables/columns + allowed joins for NL2SQL planning.

    Inputs:
      - columns_json: like columns_name_embeddings.json
          [
            {
              "db_id": "debit_card_specializing",
              "table": "customers",
              "column": "Segment",
              "col_id": 2,
              "key": "debit_card_specializing.customers.Segment",
              "db_id_vec": [...],
              "table_vec": [...],
              "column_vec": [...]
            },
            ...
          ]

      - edges_json: like edges_name_embeddings.json
          [
            {
              "db_id": "financial",
              "child_table": "account",
              "child_column": "district_id",
              "parent_table": "district",
              "parent_column": "district_id",
              "edge_key": "financial:account.district_id->district.district_id",
              "db_id_vec": [...],
              "child_table_vec": [...],
              "child_column_vec": [...],
              "parent_table_vec": [...],
              "parent_column_vec": [...]
            },
            ...
          ]

      - query_embeddings: list of embedding vectors extracted from your query
        (e.g., from extracted_graph_two_stage_embedding.json nodes[*].column_embedding or table_embedding).
        It can be 1..N vectors; we will take the MAX similarity per column across all query vecs.

    Steps:
      1) rank columns by cosine similarity (Top-K)
      2) aggregate to tables (Top-P)
      3) return payload:
         {
           "candidate_tables": [
              {
                "db_id": "...",
                "table": "customers",
                "columns": ["colA","colB", ...],          # full columns in this table
                "foreign_keys": [                         # only edges where this table participates
                    {"from": "account.district_id", "to": "district.district_id"}
                ],
                "score": 3.142                            # aggregated score for this table
              },
              ...
           ],
           "allowed_joins": [
               "account.district_id = district.district_id",
               "yearmonth.CustomerID = customers.CustomerID"
           ],
           "top_columns": [("table.col", score), ...]     # Top-K columns list
         }

    Algorithm Summary
    -----------------
    This backup implementation follows a **column-first, lightweight ranking** strategy
    without explicit-table parsing or FK-neighbor expansion:

    1) **Global column scoring**
       - For every column (optionally restricted by `db_id`), compute the cosine similarity
         between its embedding and each query embedding; keep the **maximum** over all query vectors.
       - Sort columns by this score and keep **Top-K** globally (`top_k_cols`).
       - If a column embedding is missing, the code falls back to `table_vec` when available.

    2) **Aggregate to tables (Top-P)**
       - Sum the retained column scores per `(db, table)` to obtain a **table score**.
       - Sort tables by the aggregated score and keep **Top-P** (`top_p_tables`).
       - This favors tables that have multiple high-signal columns among the Top-K.

    3) **Assemble candidates**
       - For each selected table, include **all columns** from the schema (not only Top-K),
         and attach **foreign keys** where this table is child or parent.
       - Produce `allowed_joins` by filtering FK edges whose both ends lie within the selected tables.
         Joins are deduplicated while preserving order.

    4) **Output views**
       - `candidate_tables`: full-column tables with FK lists and their aggregated scores.
       - `allowed_joins`: FK-based join strings among selected tables.
       - `top_columns`: the **global** Top-K columns list as `(db.table.column, score)` pairs
         (scores formatted with fixed precision), preserving global ranking.

    Differences vs. the primary builder
    -----------------------------------
    - No explicit-table name matching; no approximate name search.
    - No one-hop FK neighbor expansion and no connectivity-based selection.
    - The selection is driven purely by **global column similarity** followed by **per-table aggregation**,
      which is simpler, deterministic, and efficient for quick candidate generation.
    """

    def __init__(self,
                 columns_json_path: str,
                 edges_json_path: str):
        self.columns_json_path = columns_json_path
        self.edges_json_path = edges_json_path

        self._columns: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []

        # fast lookups
        self._cols_by_table: Dict[Tuple[str, str], List[str]] = defaultdict(list)  # (db_id, table) -> [col,...]
        self._all_tables_in_db: Dict[str, Set[str]] = defaultdict(set)            # db_id -> {table,...}

    # ---------- Loaders ----------

    def load(self) -> None:
        cols = FileReader.load_json(self.columns_json_path) or []
        edges = FileReader.load_json(self.edges_json_path) or []

        # normalize columns
        self._columns = []
        for item in cols:
            db = item.get("db_id")
            tbl = item.get("table")
            col = item.get("column")
            if not (db and tbl and col):
                continue
            self._columns.append(item)
            self._cols_by_table[(db, tbl)].append(col)
            self._all_tables_in_db[db].add(tbl)

        # normalize edges
        self._edges = []
        for e in edges:
            # we only need names here; vectors are not required in this phase
            if not (e.get("db_id") and e.get("child_table") and e.get("child_column")
                    and e.get("parent_table") and e.get("parent_column")):
                continue
            self._edges.append(e)

    # ---------- Core API ----------

    def build_candidates(self,
                         query_vecs: List[List[float]],
                         db_id: Optional[str] = None,
                         top_k_cols: int = 80,
                         top_p_tables: int = 5,
                         ) -> Dict[str, Any]:
        """
        Compute candidate tables/columns + allowed joins.

        Args:
          query_vecs: list of query embedding vectors (>=1)
          db_id: if set, restrict to this single database (recommended)
          top_k_cols: pick top-K columns globally by similarity
          top_p_tables: then pick top-P tables by aggregated score
        """
        assert len(query_vecs) > 0, "query_vecs must contain at least one vector"
        # 1) rank columns
        ranked_cols = self._rank_columns(query_vecs, db_id=db_id, top_k=top_k_cols)

        # 2) aggregate to tables
        table_scores, tables_sorted = self._aggregate_tables(ranked_cols, top_p=top_p_tables)

        # 3) collect full columns per selected table
        candidate_tables = []
        selected_tables_set: Set[Tuple[str, str]] = set()
        for (db, tbl) in tables_sorted:
            full_cols = sorted(set(self._cols_by_table.get((db, tbl), [])))
            # collect foreign keys (edges that touch this table)
            fks = self._collect_foreign_keys_for_table(db, tbl)
            candidate_tables.append({
                "db_id": db,
                "table": tbl,
                "columns": full_cols,
                "foreign_keys": fks,
                "score": round(table_scores[(db, tbl)], 6),
            })
            selected_tables_set.add((db, tbl))

        # 4) allowed joins among selected tables only
        allowed_joins = self._allowed_joins_among(selected_tables_set)

        # 5) package top-K columns list as ("table.col", score)
        top_columns_list = [
            (f"{db}.{tbl}.{col}", float(f"{score:.12f}"))
            for ((db, tbl, col), score) in ranked_cols
        ]

        return {
            "candidate_tables": candidate_tables,
            "allowed_joins": allowed_joins,
            "top_columns": top_columns_list
        }

    # ---------- Internals ----------

    def _rank_columns(self,
                      query_vecs: List[List[float]],
                      db_id: Optional[str],
                      top_k: int) -> List[Tuple[Tuple[str, str, str], float]]:
        """
        For all columns (optionally within one db_id), compute MAX cosine similarity
        across query_vecs; return top_k [(db, table, column), score]
        """
        scored: List[Tuple[Tuple[str, str, str], float]] = []
        for item in self._columns:
            if db_id and item.get("db_id") != db_id:
                continue
            col_vec = item.get("column_vec") or item.get("column_embedding")
            if not col_vec:
                # fallback: try table_vec when column_vec missing (rare)
                col_vec = item.get("table_vec")
                if not col_vec:
                    continue

            # MAX over all query embeddings to be robust to multiple query slots
            best = -1.0
            for qv in query_vecs:
                try:
                    s = Utils.cosine_similarity(qv, col_vec)
                except Exception:
                    s = -1.0
                if s > best:
                    best = s

            scored.append(((item["db_id"], item["table"], item["column"]), best))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _aggregate_tables(self,
                          ranked_cols: List[Tuple[Tuple[str, str, str], float]],
                          top_p: int) -> Tuple[Dict[Tuple[str, str], float], List[Tuple[str, str]]]:
        """
        Sum column scores per (db, table), pick top_p
        """
        table_scores: Dict[Tuple[str, str], float] = defaultdict(float)
        for (db, tbl, _col), score in ranked_cols:
            table_scores[(db, tbl)] += score

        tables_sorted = sorted(table_scores.keys(), key=lambda k: table_scores[k], reverse=True)[:top_p]
        return table_scores, tables_sorted

    def _collect_foreign_keys_for_table(self, db: str, table: str) -> List[Dict[str, str]]:
        """
        Return FK edges that touch this table (child or parent).
        """
        res = []
        for e in self._edges:
            if e.get("db_id") != db:
                continue
            child_tbl = e.get("child_table")
            parent_tbl = e.get("parent_table")
            if child_tbl == table or parent_tbl == table:
                res.append({
                    "from": f"{child_tbl}.{e.get('child_column')}",
                    "to": f"{parent_tbl}.{e.get('parent_column')}"
                })
        return res

    def _allowed_joins_among(self, selected: Set[Tuple[str, str]]) -> List[str]:
        """
        Filter edges where both ends are within `selected` tables; output as "a.b = c.d".
        """
        joins = []
        for e in self._edges:
            db = e.get("db_id")
            a = (db, e.get("child_table"))
            b = (db, e.get("parent_table"))
            if a in selected and b in selected:
                joins.append(f"{e['child_table']}.{e['child_column']} = {e['parent_table']}.{e['parent_column']}")
        # de-dup while preserving order
        seen = set()
        uniq = []
        for j in joins:
            if j not in seen:
                uniq.append(j)
                seen.add(j)
        return uniq


if __name__ == "__main__":
    cols_path  = "backend/src/embedding/output/columns_name_embeddings.json"
    edges_path = "backend/src/embedding/output/edges_name_embeddings.json"
    query_path = "backend/src/query/output/extracted_graph_two_stage_embedding.json"

    builder = CandidateGraphBuilder(cols_path, edges_path)
    builder.load()

    graph = FileReader.load_json(query_path) or {}
    query_vecs = []
    for n in (graph.get("nodes") or []):
        if "column_embedding" in n and n["column_embedding"]:
            query_vecs.append(n["column_embedding"])
        elif "table_embedding" in n and n["table_embedding"]:
            query_vecs.append(n["table_embedding"])

    if not query_vecs:
        print("⚠️ 没有找到 query embedding")
    else:
        payload = builder.build_candidates(
            query_vecs=query_vecs,
            db_id=None,
            top_k_cols=30,
            top_p_tables=3
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
