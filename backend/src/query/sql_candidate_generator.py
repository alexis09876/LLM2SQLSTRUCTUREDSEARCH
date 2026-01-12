import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Dict, Any

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.llm.chatgpt_client import ChatGPTClient


class SQLCandidateGenerator:
    """
    Single-call LLM executor for generating possible SQL statements (or structured planning)
    from a natural-language query, then saving the STRICT-JSON response.

    Prompts:
      - system_prompt_for_sql_candidates
      - user_prompt_template_for_sql_candidates
    """

    def __init__(self, prompts: Dict[str, str]) -> None:
        """
        Initialize the generator with an LLM client and prompt templates.

        Args:
            prompts (Dict[str, str]): Dictionary containing paths to:
                - "system_prompt_for_sql_candidates"
                - "user_prompt_template_for_sql_candidates"
        """
        self.llm_client = ChatGPTClient()

        self.system_prompt: str = FileReader.read_prompt_template_from_txt(
            prompts["system_prompt_for_sql_candidates"]
        )
        self.user_prompt_template: str = FileReader.read_prompt_template_from_txt(
            prompts["user_prompt_template_for_sql_candidates"]
        )

    def generate(self, query_text: str) -> Dict[str, Any]:
        """
        Call the LLM once to generate possible SQL candidates from a natural language query.

        The user prompt template should include a placeholder `{query}`.
        The LLM is expected to return a STRICT JSON string.

        Args:
            query_text (str): The original natural language query.

        Returns:
            Dict[str, Any]: Parsed JSON object returned by the LLM.
        """
        user_prompt = self.user_prompt_template.format(query=query_text)
        raw_response = self.llm_client.handle_chat(self.system_prompt, user_prompt)
        return json.loads(raw_response)

    def save_json(self, data: Dict[str, Any], output_path: str, indent: int = 2) -> None:
        """
        Save the given JSON-serializable object.

        Args:
            data (Dict[str, Any]): The data object to save (already parsed JSON).
            output_path (str): Full path where the JSON file will be saved.
            indent (int, optional): Number of spaces for indentation. Defaults to 2.
        """
        FileWriter.write_json(data, output_path, indent=indent)


if __name__ == "__main__":
    prompts_config = {
        "system_prompt_for_sql_candidates": "backend/src/prompt/system_sql_candidates.txt",
        "user_prompt_template_for_sql_candidates": "backend/src/prompt/user_sql_candidates_template.txt",
    }

    sample_query = "What is the ratio of customers who pay in EUR against customers who pay in CZK"
    out_path = "backend/src/query/output/sql_candidates.json"

    generator = SQLCandidateGenerator(prompts=prompts_config)
    data = generator.generate(sample_query)
    generator.save_json(data, out_path, indent=2)
    print(f"[INFO] JSON written to {out_path}")
