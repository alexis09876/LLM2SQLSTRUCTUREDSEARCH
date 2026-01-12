import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import List, Dict, Any

from backend.src.query.sql_candidate_generator import SQLCandidateGenerator 


class BatchSQLCandidateGenerator:
    """
    Batch executor that calls SQLCandidateGenerator for a list of natural-language queries
    and saves each STRICT-JSON response.

    This class wraps a single SQLCandidateGenerator instance and provides a convenient
    method to process multiple queries in one go.
    """

    def __init__(self, prompts: Dict[str, str]) -> None:
        """
        Initialize the batch generator with a single SQLCandidateGenerator.

        Args:
            prompts (Dict[str, str]): Dictionary containing paths to:
                - "system_prompt_for_sql_candidates"
                - "user_prompt_template_for_sql_candidates"
        """
        self.single = SQLCandidateGenerator(prompts=prompts)

    def generate_all(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate STRICT-JSON objects for each natural-language query without saving.

        Args:
            queries (List[str]): A list of natural-language queries.

        Returns:
            List[Dict[str, Any]]: A list of parsed JSON objects, one per query, in the same order as the input list.
        """
        results: List[Dict[str, Any]] = []
        for q in queries:
            data = self.single.generate(q)
            results.append(data)
        return results

    def generate_and_save_all(
        self,
        queries: List[str],
        output_dir: str,
        filename_prefix: str = "sql_candidates",
        indent: int = 2
    ) -> bool:
        """
        Generate and save STRICT-JSON responses for each query.

        Each output file is named as: {output_dir}/{filename_prefix}_{index:04d}.json
        where index starts from 1 and follows the input order.

        Args:
            queries (List[str]): A list of natural-language queries.
            output_dir (str): Directory where the JSON files will be saved.
            filename_prefix (str, optional): Common filename prefix for all outputs. Defaults to "sql_candidates".
            indent (int, optional): Indentation for JSON pretty-printing. Defaults to 2.

        Returns:
            bool: True if all files are saved successfully; False if any item fails.
        """
        os.makedirs(output_dir, exist_ok=True)

        all_ok = True
        for idx, q in enumerate(queries, start=1):
            try:
                data = self.single.generate(q)
                out_path = os.path.join(output_dir, f"{filename_prefix}_{idx:04d}.json")
                self.single.save_json(data, out_path, indent=indent)
            except Exception as e:
                all_ok = False
                print(f"❌ Failed on item #{idx}: {e}")
        return all_ok


if __name__ == "__main__":
    prompts_config = {
        "system_prompt_for_sql_candidates": "backend/src/prompt/system_sql_candidates.txt",
        "user_prompt_template_for_sql_candidates": "backend/src/prompt/user_sql_candidates_template.txt",
    }

    queries = [
        "What is the ratio of customers who pay in EUR against customers who pay in CZK",
        "What was the gas consumption peak month for SME customers in 2013?",
        "What is the difference in the annual average consumption of the customers with the least amount of consumption paid in CZK for 2013 between SME and LAM, LAM and KAM, and KAM and SME?"
    ]

    out_dir = "backend/src/query/output/batch"

    runner = BatchSQLCandidateGenerator(prompts=prompts_config)

    data_list = runner.generate_all(queries)
    print(f"[INFO] Generated {len(data_list)} JSON objects (not saved).")

    ok = runner.generate_and_save_all(
        queries=queries,
        output_dir=out_dir,
        filename_prefix="sql_candidates",
        indent=2
    )
    print("DEBUG batch save status:", ok)
