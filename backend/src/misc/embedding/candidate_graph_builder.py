import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import math
import re
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Set

from backend.src.data_io.file_reader import FileReader
from backend.src.utils.utils import Utils


class CandidateGraphBuilder:
    """
    A builder that selects candidate tables and columns for a query using
    schema embeddings and foreign-key (FK) relationships.

    Summary of the Algorithm (Explicit-first + 1-hop FK expansion + Fallback)
    -------------------------------------------------------------------------
    1) Explicit table matching (if `query_meta` specifies table names):
       - Extract names from `query_meta.nodes` where type=="table".
       - Case-insensitive exact matching first; if none found, fallback to
         approximate matching via substring or token-overlap.
       - For all matched explicit tables, rank their columns against query
         embeddings (max cosine per column), keep per-table Top-K columns,
         and define the table score as the sum of those kept column scores.
       - Choose the DB of the top-scoring explicit table, expand 1-hop FK
         neighbors **only in that DB** based on semantic whitelist or score threshold.
       - Merge (explicit + qualified neighbors), sort by table score, take Top-K,
         and package the final payload.

    2) Fallback (if no explicit tables are provided in the query):
       - Rank all tables by cosine(query, table representative vector),
         where the representative vector prefers `table_vec` or the mean of
         the table's column vectors.
       - Prefilter by threshold or keep a small superset, then build an
         undirected FK graph over the candidates.
       - Select tables from the highest-scoring connected component (sum of
         table scores), keep up to Top-K, re-rank columns, and package.

    What You Get
    ------------
    - `candidate_tables`:
        * All columns of each selected table (to avoid omissions)
        * Per-table FK list (incoming/outgoing)
        * Table score
        * Top matched columns with scores
    - `allowed_joins`: FK-based join conditions among the selected tables.
    - `top_columns`: Flattened, globally ranked (db.table.col, score) list.

    Typical Usage
    -------------
        builder = CandidateGraphBuilder(cols_path, edges_path)
        builder.load()
        payload = builder.build_candidates(
            query_vecs=query_vectors,     # list[list[float]], same dimension as schema vectors
            db_id=None,                   # optional: restrict to a specific DB
            top_k_tables=6,
            per_table_top_k_cols=10,
            min_table_score=0.25,
            min_col_score=0.18,
            query_meta=query_graph        # used for explicit table detection
        )

    Args:
        columns_json_path (str): Path to JSON file containing column/table embeddings.
        edges_json_path   (str): Path to JSON file containing FK edges.

    Attributes (Populated by `load()`)
    ----------------------------------
        _columns (list[dict]): All column entries with vectors.
        _edges (list[dict]): FK edges with db_id/child_table/child_column/parent_table/parent_column.
        _cols_by_table (dict[(str,str)->list[str]]): Table -> all column names.
        _all_tables_in_db (dict[str->set[str]]): DB -> tables.
        _table_to_dbs (dict[str->set[str]]): lower(table_name) -> possible DBs.
        _table_rep_vec (dict[(str,str)->list[float]]): Representative vector for each table.
        _idf (dict[str->float]): Token -> IDF, reserved for potential keyword mixing in fallback.
    """

    def __init__(self,
                 columns_json_path: str,
                 edges_json_path: str):
        """
        Initialize the builder with JSON paths.

        Args:
            columns_json_path (str): Path to columns embeddings JSON.
            edges_json_path   (str): Path to FK edges JSON.
        """
        self.columns_json_path = columns_json_path
        self.edges_json_path = edges_json_path

        self._columns: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []

        # lookups
        self._cols_by_table: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._all_tables_in_db: Dict[str, Set[str]] = defaultdict(set)
        self._table_to_dbs: Dict[str, Set[str]] = defaultdict(set)

        # built at load()
        self._table_rep_vec: Dict[Tuple[str, str], List[float]] = {}
        self._idf: Dict[str, float] = {}

    # ----------------- helpers (moved inside the class) -----------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize a string for fuzzy/approximate matching.

        - Split CamelCase (e.g., "OrderItems" -> "Order Items")
        - Split on non-alphanumeric characters
        - Lowercase and drop empties

        Args:
            text (str): Raw string.

        Returns:
            list[str]: Lowercased tokens.
        """
        if not text:
            return []
        text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)  # CamelCase split
        toks = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return [t for t in toks if t]

    # ----------------- Loaders -----------------

    def load(self) -> None:
        """
        Load schema embeddings and FK edges; build in-memory indices.

        - Reads columns and edges JSON.
        - Builds:
            * _cols_by_table, _all_tables_in_db, _table_to_dbs lookups
            * Simple IDF from column-name tokens (reserved for fallback usage)
            * Table representative vectors (via `_build_table_reps`)

        Raises:
            None. (Skips malformed rows silently.)
        """
        cols = FileReader.load_json(self.columns_json_path) or []
        edges = FileReader.load_json(self.edges_json_path) or []

        self._columns = []
        self._cols_by_table.clear()
        self._all_tables_in_db.clear()
        self._table_to_dbs.clear()

        # Build IDF for fallback keyword logic (currently reserved)
        df: Dict[str, int] = defaultdict(int)
        seen_docs: List[Set[str]] = []

        for item in cols:
            db = item.get("db_id")
            tbl = item.get("table")
            col = item.get("column")
            if not (db and tbl and col):
                continue
            vec_any = item.get("column_vec") or item.get("column_embedding") or item.get("table_vec")
            if not (isinstance(vec_any, list) and vec_any and isinstance(vec_any[0], (int, float))):
                continue

            self._columns.append(item)
            self._cols_by_table[(db, tbl)].append(col)
            self._all_tables_in_db[db].add(tbl)
            self._table_to_dbs[tbl.lower()].add(db)

            toks = set(self._tokenize(col))
            seen_docs.append(toks)

        # Simple IDF
        N = max(1, len(seen_docs))
        for toks in seen_docs:
            for t in toks:
                df[t] += 1
        self._idf = {t: math.log(1.0 + N / (1.0 + c)) for t, c in df.items()}

        # edges
        self._edges = []
        for e in edges:
            if not (e.get("db_id") and e.get("child_table") and e.get("child_column")
                    and e.get("parent_table") and e.get("parent_column")):
                continue
            self._edges.append(e)

        # build table representative vectors for fallback
        self._build_table_reps()

    def _build_table_reps(self, sample_per_table: int = 256) -> None:
        """
        Build representative vector per table (fallback-only).

        Strategy:
            - Prefer `table_vec` if present.
            - Else compute mean over up to `sample_per_table` column vectors.

        Args:
            sample_per_table (int): Max number of columns to average per table.

        Returns:
            None
        """
        by_table_cols: Dict[Tuple[str, str], List[List[float]]] = defaultdict(list)
        table_vecs: Dict[Tuple[str, str], List[float]] = {}

        for item in self._columns:
            key = (item["db_id"], item["table"])
            tv = item.get("table_vec")
            if isinstance(tv, list) and tv and isinstance(tv[0], (int, float)):
                table_vecs[key] = tv
            cv = item.get("column_vec") or item.get("column_embedding")
            if isinstance(cv, list) and cv and isinstance(cv[0], (int, float)):
                by_table_cols[key].append(cv)

        self._table_rep_vec.clear()
        tables: Set[Tuple[str, str]] = set(self._cols_by_table.keys())

        for key in tables:
            if key in table_vecs:
                self._table_rep_vec[key] = table_vecs[key]
                continue
            col_vecs = by_table_cols.get(key, [])
            if not col_vecs:
                continue
            vecs = col_vecs[:sample_per_table]
            dim = len(vecs[0])
            mean = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    mean[i] += v[i]
            count = float(len(vecs))
            self._table_rep_vec[key] = [x / count for x in mean]

    # ----------------- Public API -----------------

    def build_candidates(self,
                         query_vecs: List[List[float]],
                         db_id: Optional[str] = None,
                         top_k_tables: int = 6,
                         per_table_top_k_cols: int = 10,
                         min_table_score: float = 0.25,
                         min_col_score: float = 0.18,
                         query_meta: Optional[dict] = None
                         ) -> Dict[str, Any]:
        """
        Build the final candidate payload.

        Behavior:
            - If `query_meta` provides explicit table names:
              * Rank columns within those tables to get table scores.
              * Expand one-hop FK neighbors only within the DB of the
                top-ranked explicit table; include neighbors by semantic
                whitelist or score threshold.
              * Merge explicit + neighbors, sort by score, take Top-K.
            - Else (no explicit tables):
              * Fallback to table-rep ranking across the catalog.
              * Pick tables from the highest-scoring FK-connected component.
              * Rank columns within those tables and package.

        Args:
            query_vecs (list[list[float]]): Query embeddings (same dimension as schema vectors).
            db_id (str, optional): If provided, restricts to this DB only.
            top_k_tables (int): Max number of tables in the final payload.
            per_table_top_k_cols (int): Top-k columns to keep per table.
            min_table_score (float): Min score for table prefilter (fallback only).
            min_col_score (float): Min column similarity threshold.
            query_meta (dict, optional): Extracted graph used to read explicit tables.

        Returns:
            dict: {
                "candidate_tables": [...],
                "allowed_joins": [...],
                "top_columns": [...]
            }

        Raises:
            AssertionError: If `query_vecs` is empty.
            ValueError: If query vector dim != schema vector dim.
            RuntimeError: If no vectors exist in columns JSON.
        """
        assert query_vecs, "query_vecs must contain at least one vector"

        # 1) explicit tables (if any)
        explicit_tables = self._match_tables_from_query(query_meta)
        if explicit_tables:
            # optional DB filter
            if db_id:
                explicit_tables = [t for t in explicit_tables if t[0] == db_id]
                if not explicit_tables:
                    return {"candidate_tables": [], "allowed_joins": [], "top_columns": []}

            # 2) rank columns only within explicit tables
            top_cols_by_table, table_scores = self._rank_columns_within_tables(
                query_vecs, explicit_tables, per_table_top_k_cols, min_col_score
            )
            sorted_tables = sorted(explicit_tables, key=lambda k: table_scores.get(k, 0.0), reverse=True)[:top_k_tables]

            # 2.1 one-hop FK neighbor expansion (within the top DB)
            EXPAND_FK_NEIGHBORS = True
            NEIGHBOR_MIN_GAIN = 0.45        # min total score for a neighbor to be included
            NEIGHBOR_MAX_NUM   = 4          # maximum neighbor tables to include

            if EXPAND_FK_NEIGHBORS and sorted_tables:
                db_chosen = sorted_tables[0][0]
                base_in_db = [t for t in sorted_tables if t[0] == db_chosen]
                neighbors = self._get_fk_neighbors(db_chosen, base_in_db)

                added = 0
                for nb in neighbors:
                    if nb in sorted_tables:
                        continue
                    nb_top_cols, nb_score = self._score_single_table(
                        query_vecs, nb, per_table_top_k_cols, min_col_score
                    )
                    if self._semantic_hit_table_name_or_cols(nb, nb_top_cols) or nb_score >= NEIGHBOR_MIN_GAIN:
                        sorted_tables.append(nb)
                        top_cols_by_table[nb] = nb_top_cols
                        table_scores[nb] = nb_score
                        added += 1
                        if added >= NEIGHBOR_MAX_NUM:
                            break

            # final top-K (explicit + neighbors)
            sorted_tables = sorted(sorted_tables, key=lambda k: table_scores.get(k, 0.0), reverse=True)[:top_k_tables]

            # 3) package final payload
            payload = self._package_payload_from_tables(
                query_vecs, sorted_tables, table_scores,
                per_table_top_k_cols, min_col_score
            )
            return payload

        # -------- fallback: no explicit tables -> global table ranking --------
        ranked_tables = self._rank_tables(query_vecs, db_id=db_id)
        prefilter = [(key, s) for key, s in ranked_tables if s >= min_table_score] or ranked_tables[:max(top_k_tables * 2, 8)]
        candidate_tables = [key for key, _ in prefilter]
        table_scores_seed = dict(prefilter)

        # choose tables from the highest-scoring FK-connected component
        selected_tables = self._pick_tables_by_connectivity(candidate_tables, table_scores_seed, want=top_k_tables)
        if not selected_tables:
            selected_tables = [key for key, _ in prefilter[:top_k_tables]]

        top_cols_by_table, table_scores = self._rank_columns_within_tables(
            query_vecs, selected_tables, per_table_top_k_cols, min_col_score
        )
        payload = self._package_payload_from_tables(
            query_vecs, selected_tables, table_scores,
            per_table_top_k_cols, min_col_score
        )
        return payload

    # ----------------- Internals -----------------

    def _match_tables_from_query(self, query_meta: Optional[dict]) -> List[Tuple[str, str]]:
        """
        Extract explicit table names from `query_meta` and resolve to (db, table).

        Matching Order:
            1) Case-insensitive exact match
            2) Approximate match by substring or token overlap

        Notes:
            - Returns all matches (may include multiple DBs if same table name exists).
            - If exact matches exist, approximate matching is skipped.

        Args:
            query_meta (dict | None): Graph with nodes like
                [{"type": "table", "name": "..."}]

        Returns:
            list[(str, str)]: Unique (db_id, table) list. Order not guaranteed.
        """
        if not query_meta:
            return []
        raw_names: List[str] = []
        for n in (query_meta.get("nodes") or []):
            if (n.get("type") or "").lower() == "table":
                nm = (n.get("name") or "").strip()
                if nm:
                    raw_names.append(nm.lower())
        if not raw_names:
            return []

        # exact matches
        matched: Set[Tuple[str, str]] = set()
        for t in raw_names:
            if t in self._table_to_dbs:
                for db in self._table_to_dbs[t]:
                    matched.add((db, next(tbl for tbl in self._all_tables_in_db[db] if tbl.lower() == t)))
        if matched:
            return list(matched)

        # approximate matches: substring or token overlap
        approx: Set[Tuple[str, str]] = set()
        for t in raw_names:
            t_toks = set(self._tokenize(t))
            for db, tbls in self._all_tables_in_db.items():
                for tbl in tbls:
                    tl = tbl.lower()
                    if t in tl or tl in t:
                        approx.add((db, tbl))
                        continue
                    if t_toks and (t_toks & set(self._tokenize(tl))):
                        approx.add((db, tbl))
        return list(approx)

    def _get_fk_neighbors(self, db: str, base_tables: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Get one-hop FK neighbor tables within the same DB.

        Args:
            db (str): Target database id.
            base_tables (list[(str,str)]): Base tables to expand from (must be in `db`).

        Returns:
            list[(str,str)]: Neighbor tables directly connected via any FK.
        """
        base_set = set(base_tables)
        neighbors: Set[Tuple[str, str]] = set()
        for e in self._edges:
            if e.get("db_id") != db:
                continue
            a = (db, e.get("child_table"))
            b = (db, e.get("parent_table"))
            if a in base_set and b not in base_set:
                neighbors.add(b)
            if b in base_set and a not in base_set:
                neighbors.add(a)
        return list(neighbors)

    def _score_single_table(self,
                            query_vecs: List[List[float]],
                            key: Tuple[str, str],
                            per_table_top_k: int,
                            min_col_score: float) -> Tuple[List[Tuple[str, float]], float]:
        """
        Score a single table by ranking its columns against query vectors.

        Args:
            query_vecs (list[list[float]]): Query embeddings.
            key ((str,str)): Target table key (db, table).
            per_table_top_k (int): Columns top-k to keep.
            min_col_score (float): Per-column min score threshold.

        Returns:
            (list[(str,float)], float): (top_columns, table_score_sum)
        """
        (db, tbl) = key
        arr = [it for it in self._columns if it["db_id"] == db and it["table"] == tbl]
        scored_cols: List[Tuple[str, float]] = []
        for it in arr:
            col_vec = it.get("column_vec") or it.get("column_embedding") or it.get("table_vec")
            if not (isinstance(col_vec, list) and col_vec and isinstance(col_vec[0], (int, float))):
                continue
            s = max(Utils.cosine_similarity(qv, col_vec) for qv in query_vecs)
            if s >= min_col_score:
                scored_cols.append((it["column"], s))
        scored_cols.sort(key=lambda x: x[1], reverse=True)
        top = scored_cols[:per_table_top_k]
        score = sum(s for _, s in top)
        return top, score

    def _semantic_hit_table_name_or_cols(self,
                                         db_tbl: Tuple[str, str],
                                         top_cols: List[Tuple[str, float]]) -> bool:
        """
        Heuristic: whether a neighbor table seems semantically relevant.

        - Construct a text haystack from table name + top matched column names.
        - Check if any whitelist token appears.

        Args:
            db_tbl ((str,str)): (db, table)
            top_cols (list[(str,float)]): Top columns from scoring step.

        Returns:
            bool: True to include; False otherwise.
        """
        whitelist = {
            "consumption","usage","average",
            "currency","amount","paid",
            "customer","segment","type",
            "czk","year","date","yearmonth"
        }
        db, tbl = db_tbl
        hay = tbl.lower() + " " + " ".join(c.lower() for c, _ in top_cols)
        return any(w in hay for w in whitelist)

    def _rank_tables(self,
                     query_vecs: List[List[float]],
                     db_id: Optional[str] = None) -> List[Tuple[Tuple[str, str], float]]:
        """
        Rank all tables by cosine(query, table_rep_vector) [fallback path].

        Args:
            query_vecs (list[list[float]]): Query embeddings.
            db_id (str, optional): Restrict ranking to a single DB.

        Returns:
            list[((str,str), float)]: [((db, table), score), ...] sorted desc by score.

        Raises:
            RuntimeError: If no vectors exist in columns JSON.
            ValueError: If query vector dimension mismatches schema vectors.
        """
        any_col = next((it for it in self._columns if (it.get("column_vec") or it.get("column_embedding") or it.get("table_vec"))), None)
        if not any_col:
            raise RuntimeError("No vectors found in columns JSON.")
        dim = len(any_col.get("column_vec") or any_col.get("column_embedding") or any_col.get("table_vec"))
        for i, q in enumerate(query_vecs):
            if len(q) != dim:
                raise ValueError(f"Query embedding #{i} dim={len(q)} != expected {dim}.")

        scored: List[Tuple[Tuple[str, str], float]] = []
        for (db, tbl), tvec in self._table_rep_vec.items():
            if db_id and db != db_id:
                continue
            cos_best = max(Utils.cosine_similarity(qv, tvec) for qv in query_vecs)
            scored.append(((db, tbl), cos_best))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_columns_within_tables(self,
                                    query_vecs: List[List[float]],
                                    tables: List[Tuple[str, str]],
                                    per_table_top_k: int,
                                    min_col_score: float) -> Tuple[Dict[Tuple[str, str], List[Tuple[str, float]]],
                                                                   Dict[Tuple[str, str], float]]:
        """
        Rank columns for each given table and compute table scores.

        - For each column: score = max cosine(query_vecs, col_vec)
        - Filter by `min_col_score`
        - Keep top `per_table_top_k` columns per table
        - Table score = sum of its kept column scores

        Args:
            query_vecs (list[list[float]]): Query embeddings.
            tables (list[(str,str)]): Tables to score.
            per_table_top_k (int): Columns top-k to keep per table.
            min_col_score (float): Score threshold per column.

        Returns:
            (dict, dict):
                - top_cols_by_table: {(db,table): [(col, score), ...]}
                - table_scores: {(db,table): sum_of_top_cols}
        """
        items_by_table: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for it in self._columns:
            key = (it["db_id"], it["table"])
            items_by_table[key].append(it)

        top_cols_by_table: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        table_scores: Dict[Tuple[str, str], float] = {}

        for key in tables:
            arr = items_by_table.get(key, [])
            scored_cols: List[Tuple[str, float]] = []
            for it in arr:
                col_vec = it.get("column_vec") or it.get("column_embedding") or it.get("table_vec")
                if not (isinstance(col_vec, list) and col_vec and isinstance(col_vec[0], (int, float))):
                    continue
                s = max(Utils.cosine_similarity(qv, col_vec) for qv in query_vecs)
                if s >= min_col_score:
                    scored_cols.append((it["column"], s))
            scored_cols.sort(key=lambda x: x[1], reverse=True)
            top = scored_cols[:per_table_top_k]
            top_cols_by_table[key] = top
            table_scores[key] = sum(s for _, s in top)

        self._last_top_cols_by_table = top_cols_by_table
        return top_cols_by_table, table_scores

    def _pick_tables_by_connectivity(self,
                                     tables_sorted: List[Tuple[str, str]],
                                     table_scores: Dict[Tuple[str, str], float],
                                     want: int) -> List[Tuple[str, str]]:
        """
        Pick up to `want` tables from the highest-scoring FK-connected component.

        Steps:
            - Build an undirected graph among `tables_sorted` using FK edges.
            - BFS over components; compute component score = sum(table_scores).
            - Take component with highest score; within it, take top `want` tables by table score.

        Args:
            tables_sorted (list[(str,str)]): Candidate tables (prefiltered and sorted).
            table_scores (dict[(str,str)->float]): Table scores used to score components.
            want (int): Number of tables desired.

        Returns:
            list[(str,str)]: Selected tables.
        """
        if not tables_sorted:
            return []

        cand_set = set(tables_sorted)
        adj: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
        for e in self._edges:
            a = (e.get("db_id"), e.get("child_table"))
            b = (e.get("db_id"), e.get("parent_table"))
            if a in cand_set and b in cand_set:
                adj[a].add(b)
                adj[b].add(a)

        seen = set()
        components: List[Tuple[List[Tuple[str, str]], float]] = []  # (nodes, sum_score)
        for t in tables_sorted:
            if t in seen:
                continue
            comp = []
            q = deque([t])
            seen.add(t)
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj.get(u, []):
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            comp_score = sum(table_scores.get(x, 0.0) for x in comp)
            components.append((comp, comp_score))

        components.sort(key=lambda x: x[1], reverse=True)

        selected: List[Tuple[str, str]] = []
        for nodes, _ in components:
            nodes_sorted = sorted(nodes, key=lambda k: table_scores.get(k, 0.0), reverse=True)
            for n in nodes_sorted:
                if len(selected) < want:
                    selected.append(n)
            if len(selected) >= want:
                break

        if not selected:
            selected = tables_sorted[:want]
        return selected

    def _collect_foreign_keys_for_table(self, db: str, table: str) -> List[Dict[str, str]]:
        """
        Collect FK edges (incoming/outgoing) for a given table.

        Args:
            db (str): DB id.
            table (str): Table name.

        Returns:
            list[dict]: [{"from": "child_tbl.child_col", "to": "parent_tbl.parent_col"}, ...]
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
        Build allowed JOIN conditions among selected tables based on FK edges.

        Args:
            selected (set[(str,str)]): Final selected (db, table) pairs.

        Returns:
            list[str]: JOIN conditions like "A.a_id = B.a_id".
        """
        joins = []
        seen = set()
        for e in self._edges:
            db = e.get("db_id")
            a = (db, e.get("child_table"))
            b = (db, e.get("parent_table"))
            if a in selected and b in selected:
                j = f"{e['child_table']}.{e['child_column']} = {e['parent_table']}.{e['parent_column']}"
                if j not in seen:
                    joins.append(j)
                    seen.add(j)
        return joins

    def _package_payload_from_tables(self,
                                     query_vecs: List[List[float]],
                                     selected_tables: List[Tuple[str, str]],
                                     table_scores_seed: Dict[Tuple[str, str], float],
                                     per_table_top_k_cols: int,
                                     min_col_score: float) -> Dict[str, Any]:
        """
        Final packaging step. Ensures:
            - Each selected table includes all columns
            - Table score present (from re-scoring or seed)
            - Per-table top matched columns retained
            - Allowed joins computed
            - Global flattened top-columns list produced

        Args:
            query_vecs (list[list[float]]): Query embeddings.
            selected_tables (list[(str,str)]): Final selected tables.
            table_scores_seed (dict): Seed scores used if re-scoring absent.
            per_table_top_k_cols (int): Top-k columns per table.
            min_col_score (float): Min column score threshold.

        Returns:
            dict: See `build_candidates` return schema.
        """
        # (Re)rank columns within the selected tables
        top_cols_by_table, table_scores = self._rank_columns_within_tables(
            query_vecs, selected_tables, per_table_top_k_cols, min_col_score
        )

        candidate_tables = []
        selected_tables_set: Set[Tuple[str, str]] = set()
        for (db, tbl) in selected_tables:
            full_cols = sorted(set(self._cols_by_table.get((db, tbl), [])))
            fks = self._collect_foreign_keys_for_table(db, tbl)
            score = round(table_scores.get((db, tbl), table_scores_seed.get((db, tbl), 0.0)), 6)
            candidate_tables.append({
                "db_id": db,
                "table": tbl,
                "columns": full_cols,
                "foreign_keys": fks,
                "score": score,
                "top_matched_columns": [
                    {"column": c, "score": float(f"{s:.12f}")} for c, s in top_cols_by_table.get((db, tbl), [])
                ]
            })
            selected_tables_set.add((db, tbl))

        allowed_joins = self._allowed_joins_among(selected_tables_set)

        # Flatten global top columns
        flat_top_cols = []
        for (db, tbl), arr in top_cols_by_table.items():
            for c, s in arr:
                flat_top_cols.append((f"{db}.{tbl}.{c}", float(f"{s:.12f}")))
        flat_top_cols.sort(key=lambda x: x[1], reverse=True)

        return {
            "candidate_tables": candidate_tables,
            "allowed_joins": allowed_joins,
            "top_columns": flat_top_cols
        }


if __name__ == "__main__":
    cols_path  = "backend/src/embedding/output/columns_name_embeddings.json"
    edges_path = "backend/src/embedding/output/edges_name_embeddings.json"
    query_path = "backend/src/query/output/extracted_graph_two_stage_embedding.json"

    builder = CandidateGraphBuilder(cols_path, edges_path)
    builder.load()

    graph = FileReader.load_json(query_path) or {}

    # collect query vectors
    query_vecs = []
    for n in (graph.get("nodes") or []):
        v = n.get("column_embedding") or n.get("table_embedding")
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            query_vecs.append(v)

    if not query_vecs:
        print("⚠️ No query embeddings found")
    else:
        payload = builder.build_candidates(
            query_vecs=query_vecs,
            db_id=None,                   # restrict to a DB if needed
            top_k_tables=6,               # final top-K tables (including neighbor-expanded)
            per_table_top_k_cols=10,      # per-table top-k columns
            min_table_score=0.25,         # fallback-only threshold
            min_col_score=0.18,           # per-column threshold
            query_meta=graph              # for explicit table detection
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
