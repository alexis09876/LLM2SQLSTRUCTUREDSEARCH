import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Dict, Any, Optional

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.llm.chatgpt_client import ChatGPTClient


class QueryGraphTransformer:
    """
    Two-stage transformer:
      1) Extract ALL possible entities (nodes) from a natural-language DB query (recall-first, no omissions).
      2) Infer ALL possible relationships (edges) between the entities based on the query and extracted nodes.

    The class reads four prompt templates from txt files:
      - system_prompt_for_entity_extraction
      - user_prompt_template_for_entity_extraction
      - system_prompt_for_relation_inference
      - user_prompt_template_for_relation_inference

    It then aggregates nodes + edges into a single graph JSON.
    """

    def __init__(self, prompts: Dict[str, str]) -> None:
        """
        Initialize the transformer with an LLM client and prompt templates.

        Args:
            prompts (Dict[str, str]): Dictionary containing paths to:
                - "system_prompt_for_entity_extraction"
                - "user_prompt_template_for_entity_extraction"
                - "system_prompt_for_relation_inference"
                - "user_prompt_template_for_relation_inference"
        """
        self.llm_client = ChatGPTClient()

        self.system_prompt_entities: str = FileReader.read_prompt_template_from_txt(
            prompts["system_prompt_for_entity_extraction"]
        )
        self.user_prompt_template_entities: str = FileReader.read_prompt_template_from_txt(
            prompts["user_prompt_template_for_entity_extraction"]
        )

        self.system_prompt_relations: str = FileReader.read_prompt_template_from_txt(
            prompts["system_prompt_for_relation_inference"]
        )
        self.user_prompt_template_relations: str = FileReader.read_prompt_template_from_txt(
            prompts["user_prompt_template_for_relation_inference"]
        )

    # ---------------- Phase One：Entity Extraction ----------------

    def extract_entities(self, query_text: str) -> str:
        """
        Call the LLM to extract ALL possible entities from the query.

        The user prompt template should include a placeholder `{query}`.
        The LLM must return a STRICT JSON with at least: {"nodes":[...], "notes": "..."}.

        Args:
            query_text (str): The original natural language query.

        Returns:
            str: Raw JSON string (as returned by the LLM).
        """
        try:
            user_prompt = self.user_prompt_template_entities.format(query=query_text)
            response = self.llm_client.handle_chat(self.system_prompt_entities, user_prompt)
            return response
        except Exception as e:
            print(f"❌ Error extracting entities: {e}")
            return ""

    def _parse_entities_json(self, raw: str) -> Dict[str, Any]:
        """
        Parse entities JSON string returned by the LLM, with normalization.

        Args:
            raw (str): Raw JSON string for entities.

        Returns:
            Dict[str, Any]: Parsed entities object with guaranteed keys.
        """
        try:
            obj: Dict[str, Any] = json.loads(raw) if raw else {}
        except Exception as e:
            print(f"❌ Failed to parse entities JSON: {e}")
            obj = {}

        obj.setdefault("nodes", [])
        obj.setdefault("notes", "")
        return obj

    # ---------------- Phase II：Relation Inference ----------------

    def infer_relations(self, query_text: str, nodes_obj: Dict[str, Any]) -> str:
        """
        Call the LLM to infer ALL possible relations between nodes.

        The user prompt template should include placeholders `{query}` and `{nodes_json}`.
        The LLM must return a STRICT JSON with at least: {"edges":[...], "notes":"..."}.

        Args:
            query_text (str): Original natural language query.
            nodes_obj (Dict[str, Any]): The parsed entities object from stage one.

        Returns:
            str: Raw JSON string (as returned by the LLM).
        """
        try:
            nodes_json_str = json.dumps({"nodes": nodes_obj.get("nodes", [])}, ensure_ascii=False)
            user_prompt = self.user_prompt_template_relations.format(
                query=query_text,
                nodes_json=nodes_json_str
            )
            response = self.llm_client.handle_chat(self.system_prompt_relations, user_prompt)
            return response
        except Exception as e:
            print(f"❌ Error inferring relations: {e}")
            return ""

    def _parse_relations_json(self, raw: str) -> Dict[str, Any]:
        """
        Parse relations JSON string returned by the LLM, with normalization.

        Args:
            raw (str): Raw JSON string for relations.

        Returns:
            Dict[str, Any]: Parsed relations object with guaranteed keys.
        """
        try:
            obj: Dict[str, Any] = json.loads(raw) if raw else {}
        except Exception as e:
            print(f"❌ Failed to parse relations JSON: {e}")
            obj = {}

        obj.setdefault("edges", [])
        obj.setdefault("notes", "")
        return obj

    # ---------------- Summary：nodes + edges → graph ----------------

    def transform_query_to_graph(self, query_text: str) -> str:
        """
        Run the two-stage pipeline (entities → relations) and return a combined graph JSON string.

        Args:
            query_text (str): The user's natural language query.

        Returns:
            str: A JSON string of the final graph: {"nodes":[...], "edges":[...], "notes": "..."}.
        """
        # Stage 1: Extract entities
        raw_entities = self.extract_entities(query_text)
        ent_obj = self._parse_entities_json(raw_entities)

        # Stage 2: Infer relations
        raw_relations = self.infer_relations(query_text, ent_obj)
        rel_obj = self._parse_relations_json(raw_relations)

        # Aggregate notes（simple aggregation）
        notes_agg = " | ".join(x for x in [ent_obj.get("notes", ""), rel_obj.get("notes", "")] if x)

        final_graph = {
            "nodes": ent_obj.get("nodes", []),
            "edges": rel_obj.get("edges", []),
            "notes": notes_agg
        }
        try:
            return json.dumps(final_graph, ensure_ascii=False, indent=2)
        except Exception:
            return json.dumps({"nodes": [], "edges": [], "notes": "serialization_error"}, ensure_ascii=False)

    def transform_and_save_graph(
        self,
        query_text: str,
        output_dir: str,
        filename_prefix: str = "extracted_graph_two_stage"
    ) -> bool:
        """
        Run the two-stage pipeline and save the final graph JSON to disk.

        Args:
            query_text (str): Natural language query.
            output_dir (str): Output directory.
            filename_prefix (str): Filename prefix (without extension).

        Returns:
            bool: True if saved successfully; False otherwise.
        """
        os.makedirs(output_dir, exist_ok=True)
        graph_json_str = self.transform_query_to_graph(query_text)

        try:
            graph_obj = json.loads(graph_json_str)
        except Exception as e:
            print(f"❌ Failed to parse final graph JSON string: {e}")
            return False

        out_path = os.path.join(output_dir, f"{filename_prefix}.json")
        return FileWriter.write_json(graph_obj, out_path)


if __name__ == "__main__":
    prompts_config = {
        "system_prompt_for_entity_extraction": "backend/src/prompt/system_entities.txt",
        "user_prompt_template_for_entity_extraction": "backend/src/prompt/user_entities_template.txt",
        "system_prompt_for_relation_inference": "backend/src/prompt/system_relations.txt",
        "user_prompt_template_for_relation_inference": "backend/src/prompt/user_relations_template.txt",
    }

    sample_query = (
        "How many times was the budget in Advertisement for \"Yearly Kickoff\" meeting more than \"October Meeting\"?"
    )
    out_dir = "backend/src/query/output"

    transformer = QueryGraphTransformer(prompts=prompts_config)
    result = transformer.transform_and_save_graph(
        query_text=sample_query,
        output_dir=out_dir,
        filename_prefix="extracted_graph_two_stage"
    )
    print("DEBUG write_json returned:", result, type(result))
