import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import itertools
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

from backend.src.utils.utils import Utils
from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter


class SQLCandidateSearcher:
    """
    A searcher that matches SQL candidate embeddings against global schema embeddings.

    Workflow:
      1) Load global embedding indexes:
         - columns_name_embeddings.json  (with table_vec / column_vec)
         - edges_name_embeddings.json    (with child/parent table/column vectors)
         - optionally tables_name_embeddings.json (if available)
      2) Load one candidate *_structure_embeddings.json file:
         {
           "intent": "...",
           "tables": { "<table>": {"embedding":[...]}, ... },
           "columns": { "<table>": { "<column>": {"embedding":[...]}, ... }, ... },
           "relationships": [
             { "left":{"table":"t1","column":"c1"}, "right":{"table":"t2","column":"c2"} }
           ]
         }
      3) Perform search:
         - Table matching (Top-K per query table)
         - Column matching (Top-K per query column, with soft table scope bonus)
         - Relation verification (exact edge key or vector-based fallback)
         - Table-combination ranking (beam search across per-table Top-K)
    """

    def __init__(
        self,
        columns_index_path: str,
        edges_index_path: str,
        tables_index_path: Optional[str] = None
    ) -> None:
        """
        Initialize searcher by loading embedding indexes into memory.

        Args:
            columns_index_path (str): Path to columns_name_embeddings.json
            edges_index_path (str): Path to edges_name_embeddings.json
            tables_index_path (Optional[str]): Path to tables_name_embeddings.json,
                                               if available. If None, table embeddings
                                               will be aggregated from columns index.
        """
        self.columns_items: List[Dict[str, Any]] = FileReader.load_json(columns_index_path) or []
        self.edges_items: List[Dict[str, Any]] = FileReader.load_json(edges_index_path) or []
        self.tables_items: List[Dict[str, Any]] = FileReader.load_json(tables_index_path) or [] if tables_index_path else []

        # Build indexes
        self._build_table_index()
        self._build_column_index()
        self._build_edge_index()

    # =====================================================================
    # Public API
    # =====================================================================

    def search(
        self,
        candidate_struct_path: str,
        *,
        topk_tables: int = 3,
        topk_columns: int = 3,
        col_evidence_top_m: int = 3,
        alpha_table_prior: float = 0.4,
        beta_col_evidence: float = 0.6,
        relation_threshold: float = 0.70,
        comb_topk_per_table: int = 3,
        comb_lambda_edges: float = 0.5,
        save_report_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run table → column → relation matching and produce an adjustment report.

        Args:
            candidate_struct_path (str): Path to one *_structure_embeddings.json
            topk_tables (int): Number of top table matches to keep per query table
            topk_columns (int): Number of top column matches to keep per query column
            col_evidence_top_m (int): m used in mean_top_m for column evidence
            alpha_table_prior (float): Weight for table prior score
            beta_col_evidence (float): Weight for column evidence score
            relation_threshold (float): Cosine threshold for relation vector fallback
            comb_topk_per_table (int): Beam width (top candidates per query table)
            comb_lambda_edges (float): Weight for edge evidence in combination score
            save_report_to (Optional[str]): If set, save the report JSON to this path

        Returns:
            Dict[str, Any]: Report with matches, checks, and best table combinations
        """
        cand = FileReader.load_json(candidate_struct_path)
        if not isinstance(cand, dict):
            raise ValueError("Invalid candidate struct JSON.")

        intent = cand.get("intent", "")
        query_tables: Dict[str, Dict[str, Any]] = cand.get("tables", {}) or {}
        query_columns: Dict[str, Dict[str, Dict[str, Any]]] = cand.get("columns", {}) or {}
        relationships: List[Dict[str, Any]] = cand.get("relationships", []) or []

        # 1. Table matching (prior)
        table_matches = self._match_tables(query_tables, topk_tables)

        # 2. Column matching and evidence
        column_matches, table_col_evidence = self._match_columns(query_columns, topk_columns, col_evidence_top_m)

        # 3. Re-rank tables using column evidence
        reranked_table_matches = self._rerank_tables(table_matches, table_col_evidence, alpha_table_prior, beta_col_evidence)

        # 4. Relation verification
        relation_checks = [self._verify_relation(rel, cand, relation_threshold) for rel in relationships]

        # 5. Table combination ranking
        best_combinations = self._rank_table_combinations(
            list(query_tables.keys()),
            reranked_table_matches,
            column_matches,
            relationships,
            comb_topk_per_table,
            comb_lambda_edges
        )

        report = {
            "intent": intent,
            "table_matches": reranked_table_matches,
            "column_matches": column_matches,
            "relation_checks": relation_checks,
            "best_table_combinations": best_combinations
        }

        if save_report_to:
            FileWriter.write_json(report, save_report_to, indent=2)

        return report

    # =====================================================================
    # Internal: build indexes
    # =====================================================================

    def _build_table_index(self) -> None:
        """Build table index from tables.json or aggregate from columns.json."""
        self.table_idx: DefaultDict[str, List[Tuple[str, str, List[float]]]] = defaultdict(list)
        if self.tables_items:
            for it in self.tables_items:
                dbid, tname, tvec = str(it.get("db_id", "")), str(it.get("table", "")), it.get("table_vec", [])
                if dbid and tname and tvec:
                    self.table_idx[tname.lower()].append((dbid, tname, tvec))
        else:
            acc: DefaultDict[Tuple[str, str], List[List[float]]] = defaultdict(list)
            for it in self.columns_items:
                dbid, tname, tvec = str(it.get("db_id", "")), str(it.get("table", "")), it.get("table_vec", [])
                if dbid and tname and tvec:
                    acc[(dbid, tname)].append(tvec)
            for (dbid, tname), vecs in acc.items():
                if not vecs:
                    continue
                dim = len(vecs[0])
                mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
                self.table_idx[tname.lower()].append((dbid, tname, mean))

    def _build_column_index(self) -> None:
        """Build column index from columns.json (keyed by column name)."""
        self.column_idx: DefaultDict[str, List[Tuple[str, str, str, List[float], List[float]]]] = defaultdict(list)
        for it in self.columns_items:
            dbid, tname, cname = str(it.get("db_id", "")), str(it.get("table", "")), str(it.get("column", ""))
            cvec, tvec = it.get("column_vec", []), it.get("table_vec", [])
            if cname and cvec:
                self.column_idx[cname.lower()].append((dbid, tname, cname, cvec, tvec))

    def _build_edge_index(self) -> None:
        """Build edge exact dictionary and vector store from edges.json."""
        self.edge_exact: Dict[Tuple[str, str, str, str], int] = {}
        self.edge_vecs: List[List[float]] = []
        for i, it in enumerate(self.edges_items):
            ct, cc, pt, pc = str(it.get("child_table", "")).lower(), str(it.get("child_column", "")).lower(), \
                             str(it.get("parent_table", "")).lower(), str(it.get("parent_column", "")).lower()
            self.edge_exact[(ct, cc, pt, pc)] = i
            ctv, ccv, ptv, pcv = it.get("child_table_vec", []), it.get("child_column_vec", []), \
                                 it.get("parent_table_vec", []), it.get("parent_column_vec", [])
            self.edge_vecs.append(ctv + ccv + ptv + pcv if ctv and ccv and ptv and pcv else [])

    # =====================================================================
    # Matching helpers (tables, columns, relations, combinations)
    # =====================================================================

    def _match_tables(self, query_tables: Dict[str, Dict[str, Any]], topk: int) -> Dict[str, List[Dict[str, Any]]]:
        """Match query tables against table index, return Top-K candidates for each query table."""
        results = {}
        for q_tname, obj in query_tables.items():
            q_vec = obj.get("embedding", []) or []
            scored = []
            for items in self.table_idx.values():
                for dbid, tname, tvec in items:
                    s = Utils.cosine_similarity(q_vec, tvec)
                    if tname.lower() == q_tname.lower():
                        s += 0.03  # small name bonus
                    scored.append((s, dbid, tname))
            scored.sort(key=lambda x: x[0], reverse=True)
            results[q_tname] = [{"db_id": dbid, "table": tname, "score": s} for s, dbid, tname in scored[:topk]]
        return results

    def _match_columns(self, query_columns: Dict[str, Dict[str, Dict[str, Any]]],
                       topk: int, top_m: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
        """Match query columns against column index, compute per-table evidence score."""
        column_matches, table_col_evidence = {}, {}
        for q_tname, cols in query_columns.items():
            scores_per_col = []
            for q_cname, cobj in cols.items():
                q_cvec = cobj.get("embedding", []) or []
                fq = f"{q_tname}.{q_cname}"
                scored = []
                for items in self.column_idx.values():
                    for dbid, tname, cname, cvec, tvec in items:
                        s = Utils.cosine_similarity(q_cvec, cvec)
                        if tname.lower() == q_tname.lower():
                            s += 0.05
                        if cname.lower() == q_cname.lower():
                            s += 0.02
                        scored.append((s, dbid, tname, cname))
                scored.sort(key=lambda x: x[0], reverse=True)
                column_matches[fq] = [{"db_id": dbid, "table": tname, "column": cname, "score": s}
                                      for s, dbid, tname, cname in scored[:topk]]
                if scored:
                    scores_per_col.append(scored[0][0])
            if scores_per_col:
                scores_sorted = sorted(scores_per_col, reverse=True)
                best = scores_sorted[0]
                m = min(top_m, len(scores_sorted))
                mean_top_m = sum(scores_sorted[:m]) / m
                evidence = 0.5 * best + 0.5 * mean_top_m
            else:
                evidence = 0.0
            table_col_evidence[q_tname] = evidence
        return column_matches, table_col_evidence

    def _rerank_tables(self, table_matches: Dict[str, List[Dict[str, Any]]],
                       table_col_evidence: Dict[str, float],
                       alpha: float, beta: float) -> Dict[str, List[Dict[str, Any]]]:
        """Re-rank tables by combining prior table score with column evidence."""
        results = {}
        for q_tname, items in table_matches.items():
            ev = table_col_evidence.get(q_tname, 0.0)
            merged = []
            for it in items:
                prior = it.get("score", 0.0)
                score = alpha * prior + beta * ev
                merged.append({**it, "score_prior": prior, "score_col_ev": ev, "score": score})
            merged.sort(key=lambda x: x["score"], reverse=True)
            results[q_tname] = merged
        return results

    def _verify_relation(self, rel: Dict[str, Any], cand: Dict[str, Any], relation_threshold: float) -> Dict[str, Any]:
        """
        Verify a relation using exact edge key or vector fallback.

        Args:
            rel (Dict[str, Any]): Relation object with left/right table/column
            cand (Dict[str, Any]): Candidate JSON with embeddings
            relation_threshold (float): Cosine threshold for vector fallback

        Returns:
            Dict[str, Any]: Relation check result with fields:
                            left, right, exact_hit, cosine, passed
        """
        lt, lc = rel.get("left", {}).get("table", ""), rel.get("left", {}).get("column", "")
        rt, rc = rel.get("right", {}).get("table", ""), rel.get("right", {}).get("column", "")
        exact = (lt.lower(), lc.lower(), rt.lower(), rc.lower()) in self.edge_exact
        cos_sim = 0.0
        if not exact:
            ltv, rtv = cand["tables"].get(lt, {}).get("embedding", []), cand["tables"].get(rt, {}).get("embedding", [])
            lcv, rcv = cand["columns"].get(lt, {}).get(lc, {}).get("embedding", []), cand["columns"].get(rt, {}).get(rc, {}).get("embedding", [])
            if ltv and lcv and rtv and rcv:
                q = ltv + lcv + rtv + rcv
                cos_sim = max(Utils.cosine_similarity(q, ev) for ev in self.edge_vecs if ev and len(ev) == len(q))
        return {"left": f"{lt}.{lc}", "right": f"{rt}.{rc}", "exact_hit": exact,
                "cosine": cos_sim, "passed": exact or cos_sim >= relation_threshold}

    def _rank_table_combinations(self, query_tables: List[str],
                                 table_matches: Dict[str, List[Dict[str, Any]]],
                                 column_matches: Dict[str, List[Dict[str, Any]]],
                                 relationships: List[Dict[str, Any]],
                                 topk_per_table: int, lambda_edges: float) -> List[Dict[str, Any]]:
        """Rank combinations of table matches using table scores and edge evidence."""
        choices = [[(x["db_id"], x["table"], x["score"]) for x in table_matches.get(qt, [])[:topk_per_table]] or [("", "", 0.0)]
                   for qt in query_tables]
        combos = list(itertools.product(*choices))
        out = []
        for combo in combos:
            mapping = {qt: combo[i][1] for i, qt in enumerate(query_tables)}
            table_scores = [combo[i][2] for i in range(len(query_tables))]
            mean_table_score = sum(table_scores) / max(1, len(table_scores))
            edge_scores = []
            for rel in relationships:
                lt, lc = rel["left"]["table"], rel["left"]["column"]
                rt, rc = rel["right"]["table"], rel["right"]["column"]
                tgt_lt, tgt_rt = mapping.get(lt, lt), mapping.get(rt, rt)
                exact = (tgt_lt.lower(), lc.lower(), tgt_rt.lower(), rc.lower()) in self.edge_exact
                edge_scores.append(1.0 if exact else 0.0)
            mean_edge_score = sum(edge_scores) / max(1, len(edge_scores)) if edge_scores else 0.0
            score = mean_table_score + lambda_edges * mean_edge_score
            out.append({"mapping": mapping, "score": round(score, 6),
                        "mean_table_score": round(mean_table_score, 6),
                        "mean_edge_score": round(mean_edge_score, 6)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:min(10, len(out))]


if __name__ == "__main__":
    columns_index = "backend/src/embedding/output/columns_name_embeddings.json"
    edges_index = "backend/src/embedding/output/edges_name_embeddings.json"
    candidate_struct = "backend/src/query/output/batch/sql_candidates_0001_structure_embeddings.json"

    searcher = SQLCandidateSearcher(columns_index, edges_index)
    report = searcher.search(candidate_struct, save_report_to=os.path.join(os.path.dirname(candidate_struct), "sql_candidates_0001_search_report.json"))
    print("[INFO] Done. Report keys:", list(report.keys()))
