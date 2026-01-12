import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import glob
from typing import List, Dict, Any, Optional, Tuple

from backend.src.twophase.sql_candidate_searcher import SQLCandidateSearcher
from backend.src.data_io.file_writer import FileWriter


class BatchSQLCandidateSearcher:
    """
    Batch runner that applies SQLCandidateSearcher to all candidate structure-embedding
    JSON files in a directory, saving a per-file search report.

    For each input file matching the glob pattern (e.g., '*_structure_embeddings.json'),
    the class produces a report JSON next to the input (or under a specified output dir).
    """

    def __init__(
        self,
        columns_index_path: str,
        edges_index_path: str,
        tables_index_path: Optional[str] = None
    ) -> None:
        """
        Initialize the batch runner with shared indexes.

        Args:
            columns_index_path (str): Path to columns_name_embeddings.json.
            edges_index_path (str): Path to edges_name_embeddings.json.
            tables_index_path (Optional[str]): Path to tables_name_embeddings.json (if available).
        """
        self.searcher = SQLCandidateSearcher(
            columns_index_path=columns_index_path,
            edges_index_path=edges_index_path,
            tables_index_path=tables_index_path
        )

    def process_dir(
        self,
        input_dir: str,
        *,
        pattern: str = "*_structure_embeddings.json",
        output_dir: Optional[str] = None,
        report_suffix: str = "_search_report.json",
        topk_tables: int = 3,
        topk_columns: int = 3,
        col_evidence_top_m: int = 3,
        alpha_table_prior: float = 0.4,
        beta_col_evidence: float = 0.6,
        relation_threshold: float = 0.70,
        comb_topk_per_table: int = 3,
        comb_lambda_edges: float = 0.5
    ) -> List[Tuple[str, Optional[str], bool, Optional[str]]]:
        """
        Run search on all matching files in a directory.

        Args:
            input_dir (str): Directory containing *_structure_embeddings.json files.
            pattern (str, optional): Glob pattern for input files. Defaults to '*_structure_embeddings.json'.
            output_dir (Optional[str], optional): If set, save reports under this directory;
                                                  otherwise save next to each input file.
            report_suffix (str, optional): Suffix for the report filename. Defaults to '_search_report.json'.
            topk_tables (int): Top-K table matches per query table.
            topk_columns (int): Top-K column matches per query column.
            col_evidence_top_m (int): m for mean_top_m in column evidence.
            alpha_table_prior (float): Weight for prior table score.
            beta_col_evidence (float): Weight for column evidence.
            relation_threshold (float): Cosine threshold for relation vector fallback.
            comb_topk_per_table (int): Beam width for table combinations.
            comb_lambda_edges (float): Weight for edge evidence.

        Returns:
            List[Tuple[str, Optional[str], bool, Optional[str]]]:
                A per-file summary list of (input_path, report_path, ok, error_message).
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        input_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
        results: List[Tuple[str, Optional[str], bool, Optional[str]]] = []

        for in_path in input_paths:
            try:
                base = os.path.splitext(os.path.basename(in_path))[0]
                out_dir = output_dir if output_dir else os.path.dirname(in_path)
                report_path = os.path.join(out_dir, f"{base}{report_suffix}")

                self.searcher.search(
                    candidate_struct_path=in_path,
                    topk_tables=topk_tables,
                    topk_columns=topk_columns,
                    col_evidence_top_m=col_evidence_top_m,
                    alpha_table_prior=alpha_table_prior,
                    beta_col_evidence=beta_col_evidence,
                    relation_threshold=relation_threshold,
                    comb_topk_per_table=comb_topk_per_table,
                    comb_lambda_edges=comb_lambda_edges,
                    save_report_to=report_path
                )
                results.append((in_path, report_path, True, None))
                print(f"[INFO] Processed: {in_path} -> {report_path}")
            except Exception as e:
                results.append((in_path, None, False, str(e)))
                print(f"❌ Failed: {in_path} | {e}")

        return results


if __name__ == "__main__":
    columns_index = "backend/src/embedding/output/columns_name_embeddings.json"
    edges_index = "backend/src/embedding/output/edges_name_embeddings.json"

    input_dir = "backend/src/query/output/batch"
    output_dir = None  # or set to a directory like "backend/src/query/output/search_reports"

    runner = BatchSQLCandidateSearcher(
        columns_index_path=columns_index,
        edges_index_path=edges_index,
        tables_index_path=None
    )

    summary = runner.process_dir(
        input_dir=input_dir,
        pattern="*_structure_embeddings.json",
        output_dir=output_dir,
        report_suffix="_search_report.json",
        topk_tables=3,
        topk_columns=3,
        col_evidence_top_m=3,
        alpha_table_prior=0.4,
        beta_col_evidence=0.6,
        relation_threshold=0.70,
        comb_topk_per_table=3,
        comb_lambda_edges=0.5
    )

    total = len(summary)
    ok = sum(1 for _, _, s, _ in summary if s)
    fail = total - ok
    print(f"[SUMMARY] total={total}, ok={ok}, fail={fail}")
    for in_path, out_path, s, err in summary:
        if not s:
            print(f"  - Fail: {in_path} | {err}")
