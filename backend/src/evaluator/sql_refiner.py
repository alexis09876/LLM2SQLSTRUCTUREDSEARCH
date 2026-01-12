import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Any, Dict, List, Optional

from backend.src.llm.chatgpt_client import ChatGPTClient
from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter


class SQLRefiner:
    """
    Single-file SQL refiner that:
      - Reads the original intent and SQL candidates from a JSON file.
      - Reads a search report JSON (tables/columns matches and relation checks).
      - Calls the LLM once (system + user prompts) to produce a revised SQL and reasoning.
      - Saves the STRICT-JSON LLM response to an output path.

    This class assumes the LLM will return STRICT JSON as enforced by your prompts.
    """

    def __init__(
        self,
        prompts: Dict[str, str],
        chat_client: Optional[ChatGPTClient] = None,
        max_json_chars: int = 120000
    ) -> None:
        """
        Initialize the refiner with an LLM client and prompt templates.

        Args:
            prompts (Dict[str, str]): Dictionary containing paths to:
                - "system_prompt_for_sql_refine"
                - "user_prompt_template_for_sql_refine"
            chat_client (ChatGPTClient, optional): Reusable LLM client instance.
            max_json_chars (int): Max characters preserved for large JSON blocks when embedding them into prompts (excess is truncated).
        """
        self.chat_client = chat_client or ChatGPTClient(model_name="gpt-4o-mini", temperature=0.0)
        self.max_json_chars = max_json_chars

        self.system_prompt: str = FileReader.read_prompt_template_from_txt(
            prompts["system_prompt_for_sql_refine"]
        )
        self.user_prompt_template: str = FileReader.read_prompt_template_from_txt(
            prompts["user_prompt_template_for_sql_refine"]
        )

    # ---------------- internal helpers ----------------

    def _compact_json(self, obj: Any) -> str:
        """
        Serialize an object into a compact JSON string with a character cap.

        Args:
            obj (Any): JSON-serializable object.

        Returns:
            str: Compact JSON string truncated with an ellipsis if oversized.
        """
        try:
            s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            s = json.dumps(str(obj), ensure_ascii=False)
        if len(s) > self.max_json_chars:
            s = s[: self.max_json_chars] + "...(truncated)"
        return s

    def _normalize_candidates(self, raw_candidates: Any) -> List[str]:
        """
        Normalize various candidate representations into a list of SQL strings.

        Args:
            raw_candidates (Any): The 'candidates' value from the original JSON.

        Returns:
            List[str]: List of SQL candidate strings.
        """
        if not raw_candidates:
            return []
        if isinstance(raw_candidates, list):
            out: List[str] = []
            for c in raw_candidates:
                if isinstance(c, str):
                    out.append(c.strip())
                elif isinstance(c, dict) and "sql" in c and isinstance(c["sql"], str):
                    out.append(c["sql"].strip())
            return out
        if isinstance(raw_candidates, dict) and "sql" in raw_candidates and isinstance(raw_candidates["sql"], str):
            return [raw_candidates["sql"].strip()]
        return []

    def _build_user_prompt(
        self,
        intent: str,
        candidates_sql: List[str],
        search_report: Dict[str, Any],
        dialect: Optional[str]
    ) -> str:
        """
        Build the user prompt by injecting intent, normalized SQL candidates, search report, and optional dialect.

        Placeholders required in template:
            {user_query}, {original_sql_json}, {search_report_json}, {dialect}

        Args:
            intent (str): Natural-language query intent.
            candidates_sql (List[str]): List of SQL candidate strings.
            search_report (Dict[str, Any]): Search report JSON produced by your searcher.
            dialect (Optional[str]): SQL dialect hint.

        Returns:
            str: Rendered user prompt string.
        """
        original_sql_json = self._compact_json({"candidates": candidates_sql})
        report_json = self._compact_json(search_report)
        return self.user_prompt_template.format(
            user_query=intent or "",
            original_sql_json=original_sql_json,
            search_report_json=report_json,
            dialect=str(dialect or "")
        )

    # ---------------- public API ----------------

    def refine_from_files(
        self,
        original_json_path: str,
        search_report_path: str,
        out_path: str,
        *,
        dialect: Optional[str] = None,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Read the original intent + SQL candidates and the search report from disk, run the LLM refinement,
        and write the STRICT-JSON result to the output file.

        Args:
            original_json_path (str): Path to original JSON containing "intent" and "candidates".
            search_report_path (str): Path to the search report JSON.
            out_path (str): Destination file path for the refined STRICT-JSON output.
            dialect (Optional[str]): SQL dialect hint (e.g., "postgresql", "mysql").
            max_tokens (int): LLM response token cap.

        Returns:
            Dict[str, Any]: The parsed STRICT-JSON response from the LLM.
        """
        original_obj = FileReader.load_json(original_json_path) or {}
        search_report = FileReader.load_json(search_report_path) or {}

        intent: str = str(original_obj.get("intent", "") or "")
        candidates_sql: List[str] = self._normalize_candidates(original_obj.get("candidates"))

        user_prompt = self._build_user_prompt(
            intent=intent,
            candidates_sql=candidates_sql,
            search_report=search_report,
            dialect=dialect
        )

        raw = self.chat_client.handle_chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )

        result_obj: Dict[str, Any] = json.loads(raw)
        FileWriter.write_json(result_obj, out_path, indent=2)
        return result_obj


if __name__ == "__main__":
    prompts_cfg = {
        "system_prompt_for_sql_refine": "backend/src/prompt/system_sql_refine.txt",
        "user_prompt_template_for_sql_refine": "backend/src/prompt/user_sql_refine_template.txt",
    }

    original_json_path = "backend/src/query/output/batch/sql_candidates_0001.json"
    search_report_path = "backend/src/query/output/batch/sql_candidates_0001_structure_embeddings_search_report.json"

    refined_out_path = "backend/src/query/output/batch/sql_candidates_0001_sql_refinement.json"

    refiner = SQLRefiner(prompts=prompts_cfg, chat_client=ChatGPTClient())

    try:
        refined = refiner.refine_from_files(
            original_json_path=original_json_path,
            search_report_path=search_report_path,
            out_path=refined_out_path,
            dialect="mysql",
            max_tokens=3000
        )
        print("[INFO] Refined JSON saved to:", refined_out_path)
        # Optional: print a quick peek
        print(json.dumps({
            "intent": refined.get("intent"),
            "dialect": refined.get("dialect"),
            "final_sql_preview": (refined.get("final_sql") or "")[:180]
        }, ensure_ascii=False, indent=2))
    except Exception as e:
        print("[ERROR] SQL refinement failed:", str(e))
