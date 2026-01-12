import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import glob
from typing import List, Optional, Tuple

from backend.src.query.sql_structure_embedding_builder import SQLStructureEmbeddingBuilder 


class BatchSQLStructureEmbeddingBuilder:
    """
    Batch processor that scans a directory for LLM candidate JSON files and, for each file,
    builds K:V embeddings for tables/columns and normalized column↔column relationships
    by delegating to SQLStructureEmbeddingBuilder.

    Each input file is processed independently; failures on one file do not stop others.
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """
        Initialize the batch builder with the underlying single-file builder.

        Args:
            model_name (str): Embedding model name used by SQLStructureEmbeddingBuilder.
                              Defaults to 'text-embedding-3-small'.
        """
        self.single = SQLStructureEmbeddingBuilder(model_name=model_name)

    def process_dir(
        self,
        input_dir: str,
        pattern: str = "*.json",
        output_dir: Optional[str] = None,
        output_suffix: str = "_structure_embeddings.json"
    ) -> List[Tuple[str, Optional[str], bool, Optional[str]]]:
        """
        Process all JSON files in a directory (matching pattern). For each input file, produce
        an output JSON that contains:
          - tables/columns embeddings (K:V)
          - normalized relationships
          - intent/source_notes passthrough

        If output_dir is None, outputs are placed alongside the input with `output_suffix`.
        Otherwise, outputs go to `output_dir` with the same base filename + `output_suffix`.

        Args:
            input_dir (str): Directory containing LLM candidate JSONs.
            pattern (str, optional): Glob pattern for input files. Defaults to "*.json".
            output_dir (Optional[str], optional): Directory to write outputs. If None, write next to inputs.
            output_suffix (str, optional): Suffix appended to the input basename for output files.

        Returns:
            List[Tuple[str, Optional[str], bool, Optional[str]]]:
                A list of records (input_path, output_path, ok, error_message).
                - ok=True if processed successfully; otherwise False with error_message.
        """
        os.makedirs(output_dir, exist_ok=True) if output_dir else None

        results: List[Tuple[str, Optional[str], bool, Optional[str]]] = []
        input_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))

        for in_path in input_paths:
            try:
                base = os.path.splitext(os.path.basename(in_path))[0]
                out_dir = output_dir if output_dir else os.path.dirname(in_path)
                out_path = os.path.join(out_dir, f"{base}{output_suffix}")

                self.single.process(in_path, out_path)
                results.append((in_path, out_path, True, None))
                print(f"[INFO] Processed: {in_path} -> {out_path}")
            except Exception as e:
                results.append((in_path, None, False, str(e)))
                print(f"❌ Failed: {in_path} | {e}")

        return results


if __name__ == "__main__":
    input_dir = "backend/src/query/output/batch"
    output_dir = None
    pattern = "*.json"

    runner = BatchSQLStructureEmbeddingBuilder(model_name="text-embedding-3-small")
    summary = runner.process_dir(
        input_dir=input_dir,
        pattern=pattern,
        output_dir=output_dir,
        output_suffix="_structure_embeddings.json"
    )

    total = len(summary)
    ok = sum(1 for _, _, s, _ in summary if s)
    fail = total - ok
    print(f"[SUMMARY] total={total}, ok={ok}, fail={fail}")
    if fail:
        for in_path, _, s, err in summary:
            if not s:
                print(f"  - Fail: {in_path} | {err}")
