import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, List, Optional
from backend.src.llm.chatgpt_client import ChatGPTClient
from backend.src.evaluator.sql_refiner import SQLRefiner


class BatchSQLRefiner:
    """
    Batch wrapper around SQLRefiner for a single directory layout.

    Expected filenames in `work_dir` (default pattern set for your current tree):
      - candidates:  sql_candidates_0001.json
      - search rep.: sql_candidates_0001_structure_embeddings_search.json
      - output:      sql_candidates_0001_sql_refinement.json
    """

    def __init__(
        self,
        prompts: Dict[str, str],
        work_dir: str,
        dialect: str = "",
        chat_client: Optional[ChatGPTClient] = None,
        candidate_glob: str = "sql_candidates_*.json",
        search_tpl: str = "{stem}_structure_embeddings_search.json",
        output_tpl: str = "{stem}_sql_refinement.json",
    ) -> None:
        """
        Initialize the batch refiner.

        Args:
            prompts (Dict[str, str]): Paths to system/user prompt templates for SQLRefiner.
            work_dir (str): Directory containing all batch jsons.
            dialect (str, optional): SQL dialect hint.
            chat_client (ChatGPTClient, optional): Shared LLM client.
            candidate_glob (str): Glob to find candidate files.
            search_tpl (str): Format template to locate search report by stem.
            output_tpl (str): Format template to name refined output by stem.
        """
        self.refiner = SQLRefiner(prompts=prompts, chat_client=chat_client or ChatGPTClient())
        self.work_dir = work_dir
        self.dialect = dialect
        self.candidate_glob = candidate_glob
        self.search_tpl = search_tpl
        self.output_tpl = output_tpl

    def _is_plain_candidate(self, path: str) -> bool:
        """
        Filter out non-plain candidate files (e.g., *_structure_*.json, *_sql_refinement.json).
        """
        name = os.path.basename(path)
        # 保留类似 sql_candidates_0001.json；剔除 *_structure_*.json 和 *_sql_refinement.json
        return ("_structure_" not in name) and ("_sql_refinement" not in name)

    def run(self) -> List[str]:
        """
        Iterate over candidate files, pair with search report, run SQLRefiner, and write outputs.

        Returns:
            List[str]: Written refined file paths.
        """
        written: List[str] = []

        cand_paths = [
            p for p in glob.glob(os.path.join(self.work_dir, self.candidate_glob))
            if self._is_plain_candidate(p)
        ]
        cand_paths.sort()

        if not cand_paths:
            print(f"[WARN] No candidate files matched in {self.work_dir}")
            return written

        for cand_path in cand_paths:
            stem = os.path.splitext(os.path.basename(cand_path))[0]  # e.g., sql_candidates_0001
            search_name = self.search_tpl.format(stem=stem)
            out_name = self.output_tpl.format(stem=stem)

            search_path = os.path.join(self.work_dir, search_name)
            out_path = os.path.join(self.work_dir, out_name)

            if not os.path.exists(search_path):
                print(f"[SKIP] Search report missing for {stem}: {search_name}")
                continue

            try:
                self.refiner.refine_from_files(
                    original_json_path=cand_path,
                    search_report_path=search_path,
                    out_path=out_path,
                    dialect=self.dialect
                )
                written.append(out_path)
                print(f"[OK] {os.path.basename(cand_path)} -> {out_name}")
            except Exception as e:
                print(f"[ERROR] {stem}: {e}")

        return written


if __name__ == "__main__":
    prompts_cfg = {
        "system_prompt_for_sql_refine": "backend/src/prompt/system_sql_refine.txt",
        "user_prompt_template_for_sql_refine": "backend/src/prompt/user_sql_refine_template.txt",
    }

    work_dir = "backend/src/query/output/batch"

    runner = BatchSQLRefiner(
        prompts=prompts_cfg,
        work_dir=work_dir,
        dialect="mysql",
        candidate_glob="sql_candidates_*.json",
        search_tpl="{stem}_structure_embeddings_search_report.json",
        output_tpl="{stem}_sql_refinement.json",
    )
    runner.run()
