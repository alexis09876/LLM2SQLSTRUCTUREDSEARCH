import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Any, Dict, Optional, Union

from backend.src.llm.chatgpt_client import ChatGPTClient
from backend.src.embedding.candidate_graph_builder import CandidateGraphBuilder
from backend.src.data_io.file_reader import FileReader


class QueryPayloadEvaluator:
    """
    LLM-based evaluator for two modes:
      (A) With payload: reconcile user_query + extracted_graph + candidate payload → SQL
      (B) Query-only:   only user_query → SQL

    Responsibilities:
      1) Read extracted_graph_two_stage.json via FileReader (mode A).
      2) Load system/user prompt templates from file paths for both modes.
      3) Construct prompts and call ChatGPTClient.handle_chat.
      4) Return raw LLM output string (STRICT JSON expected by your prompts).
    """

    def __init__(
        self,
        prompts: Dict[str, str],
        chat_client: Optional[ChatGPTClient] = None,
        max_json_chars: int = 120000
    ) -> None:
        """
        Initialize the evaluator with LLM client and prompt templates loaded from file paths.

        Args:
            prompts (Dict[str, str]): Dictionary containing paths to:
                # Mode A (with payload)
                - "system_prompt_for_sql_w_payload"
                - "user_prompt_template_for_sql_w_payload"
                # Mode B (query only)
                - "system_prompt_for_sql_query_only"
                - "user_prompt_template_for_sql_query_only"
            chat_client (ChatGPTClient, optional): Reusable LLM client instance.
            max_json_chars (int): Max characters kept for each JSON block to avoid
                overly large prompts. Extra content will be truncated with an ellipsis.
        """
        self.chat_client = chat_client or ChatGPTClient(model_name="gpt-4o-mini", temperature=0.0)
        self.max_json_chars = max_json_chars

        # ----- Mode A: with payload -----
        self.system_prompt_w_payload: Optional[str] = FileReader.read_prompt_template_from_txt(
            prompts.get("system_prompt_for_sql_w_payload", "")
        ) if "system_prompt_for_sql_w_payload" in prompts else None

        self.user_prompt_template_w_payload: Optional[str] = FileReader.read_prompt_template_from_txt(
            prompts.get("user_prompt_template_for_sql_w_payload", "")
        ) if "user_prompt_template_for_sql_w_payload" in prompts else None

        # ----- Mode B: query only -----
        self.system_prompt_query_only: Optional[str] = FileReader.read_prompt_template_from_txt(
            prompts.get("system_prompt_for_sql_query_only", "")
        ) if "system_prompt_for_sql_query_only" in prompts else None

        self.user_prompt_template_query_only: Optional[str] = FileReader.read_prompt_template_from_txt(
            prompts.get("user_prompt_template_for_sql_query_only", "")
        ) if "user_prompt_template_for_sql_query_only" in prompts else None

    def _to_compact_json(self, obj: Union[Dict[str, Any], Any]) -> str:
        """
        Safely serialize an object to compact JSON string, with a character cap.

        Args:
            obj (Any): Any JSON-serializable object.

        Returns:
            str: Compact JSON (minified) and optionally truncated with an ellipsis.
        """
        try:
            s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            s = json.dumps(str(obj), ensure_ascii=False)

        if len(s) > self.max_json_chars:
            s = s[: self.max_json_chars] + "...(truncated)"
        return s

    def _build_user_prompt_w_payload(
        self,
        user_query: str,
        extracted_graph: Optional[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> str:
        """
        Build the user prompt for Mode A (with payload).
        Placeholders required in template: {user_query}, {extracted_graph_json}, {candidate_payload_json}
        """
        if not self.user_prompt_template_w_payload:
            raise ValueError("user_prompt_template_for_sql_w_payload is not loaded or path missing.")

        extracted_json = self._to_compact_json(
            extracted_graph or {"warning": "extracted_graph_two_stage.json not found or invalid"}
        )
        payload_json = self._to_compact_json(payload)

        # Provide both canonical and alias keys to avoid KeyError in templates
        mapping = {
            "user_query": user_query,
            "extracted_graph_json": extracted_json,
            "candidate_payload_json": payload_json,
            # aliases in case template uses different names
            "payload_json": payload_json,
            "extracted_graph_two_stage_json": extracted_json,
        }

        # Use format_map to avoid KeyError for stray placeholders
        from collections import defaultdict
        return self.user_prompt_template_w_payload.format_map(defaultdict(str, mapping))

    def _build_user_prompt_query_only(self, user_query: str) -> str:
        """
        Build the user prompt for Mode B (query only).
        Placeholder required in template: {user_query}
        """
        if not self.user_prompt_template_query_only:
            raise ValueError("user_prompt_template_for_sql_query_only is not loaded or path missing.")
        return self.user_prompt_template_query_only.format(user_query=user_query)

    # ---------------- Public APIs ----------------

    def evaluate(
        self,
        user_query: str,
        extracted_graph_path: str,
        payload: Dict[str, Any],
        max_tokens: int = 4000
    ) -> str:
        """
        Mode A: Run evaluation with payload by composing prompts and calling the LLM.

        Args:
            user_query (str): The original natural-language question.
            extracted_graph_path (str): Path to 'extracted_graph_two_stage.json'.
            payload (dict): CandidateGraphBuilder payload (tables/columns/joins/scores...).
            max_tokens (int, optional): LLM response token cap.

        Returns:
            str: Raw LLM response string (expected to be STRICT JSON per contract).
        """
        if not self.system_prompt_w_payload:
            raise ValueError("system_prompt_for_sql_w_payload is not loaded or path missing.")

        # 1) Load extracted_graph_two_stage.json
        extracted_graph = FileReader.load_json(extracted_graph_path)

        # 2) Build user prompt
        user_prompt = self._build_user_prompt_w_payload(
            user_query=user_query,
            extracted_graph=extracted_graph,
            payload=payload
        )

        # 3) Call LLM
        response = self.chat_client.handle_chat(
            system_prompt=self.system_prompt_w_payload,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )
        return response

    def evaluate_query_only(
        self,
        user_query: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Mode B: Run evaluation using only the natural-language query.

        Args:
            user_query (str): The original natural-language question.
            max_tokens (int, optional): LLM response token cap.

        Returns:
            str: Raw LLM response string (expected to be STRICT JSON per your query-only prompt).
        """
        if not self.system_prompt_query_only:
            raise ValueError("system_prompt_for_sql_query_only is not loaded or path missing.")

        # Build user prompt with only {user_query}
        user_prompt = self._build_user_prompt_query_only(user_query=user_query)

        # Call LLM
        response = self.chat_client.handle_chat(
            system_prompt=self.system_prompt_query_only,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )
        return response


if __name__ == "__main__":
    cols_path  = "backend/src/embedding/output/columns_name_embeddings.json"
    edges_path = "backend/src/embedding/output/edges_name_embeddings.json"
    query_path_embed = "backend/src/query/output/extracted_graph_two_stage_embedding.json"
    query_path_struct = "backend/src/query/output/extracted_graph_two_stage.json"

    builder = CandidateGraphBuilder(cols_path, edges_path)
    builder.load()

    graph = FileReader.load_json(query_path_embed) or {}

    # collect query embeddings
    query_vecs = []
    for node in (graph.get("nodes") or []):
        v = node.get("column_embedding") or node.get("table_embedding")
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            query_vecs.append(v)

    if not query_vecs:
        print("⚠️ No query embeddings found in extracted_graph_two_stage_embedding.json")
        payload = {}
    else:
        payload = builder.build_candidates(
            query_vecs=query_vecs,
            db_id=None,
            top_k_tables=6,
            per_table_top_k_cols=10,
            min_table_score=0.25,
            min_col_score=0.18,
            query_meta=graph
        )

    print("=== Candidate Payload ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Prompts configuration
    prompts = {
        # Mode A (with payload)
        "system_prompt_for_sql_w_payload": "backend/src/prompt/system_prompt_for_sql_w_payload.txt",
        "user_prompt_template_for_sql_w_payload": "backend/src/prompt/user_prompt_template_for_sql_w_payload.txt",
        # Mode B (query only)
        "system_prompt_for_sql_query_only": "backend/src/prompt/system_prompt_for_sql_query_only.txt",
        "user_prompt_template_for_sql_query_only": "backend/src/prompt/user_prompt_template_for_sql_query_only.txt",
    }

    evaluator = QueryPayloadEvaluator(prompts=prompts, chat_client=ChatGPTClient())

    user_query = (
        "How many times was the budget in Advertisement for \"Yearly Kickoff\" meeting more than \"October Meeting\"?"
    )

    # ----- Mode A: with payload -----
    try:
        result_with = evaluator.evaluate(
            user_query=user_query,
            extracted_graph_path=query_path_struct,
            payload=payload,
            max_tokens=3000
        )
        print("\n=== LLM Evaluation Output (with payload) ===")
        print(result_with)
    except Exception as e:
        print(f"[Error] Evaluation (with payload) failed: {e}")

    # ----- Mode B: query only -----
    try:
        result_q_only = evaluator.evaluate_query_only(
            user_query=user_query,
            max_tokens=3000
        )
        print("\n=== LLM Evaluation Output (query only) ===")
        print(result_q_only)
    except Exception as e:
        print(f"[Error] Evaluation (query only) failed: {e}")
