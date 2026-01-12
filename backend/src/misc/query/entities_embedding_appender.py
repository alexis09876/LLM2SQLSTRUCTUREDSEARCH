import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Any, Dict, List, Tuple

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.embedding.embedding_client import EmbeddingClient

class EntitiesEmbeddingAppender:
    """
    A processor for appending embeddings to column nodes within a JSON schema.

    This class extracts column names and table names from nodes in the input JSON,
    generates embeddings for each occurrence (no deduplication), and appends the
    embeddings back into the corresponding column nodes.
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """
        Initialize the EntitiesEmbeddingAppender.

        Args:
            model_name (str): The embedding model to be used.
                              Defaults to 'text-embedding-3-small'.
        """
        self.embedder = EmbeddingClient(model_name=model_name)

    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process the input JSON file, generate embeddings for column and table names,
        and write the updated JSON to the output file.

        Args:
            input_path (str): Path to the input JSON file containing "nodes".
            output_path (str): Path to save the output JSON with appended embeddings.

        Returns:
            Dict[str, Any]: The updated JSON data with embeddings appended.
        """
        data = FileReader.load_json(input_path)
        if data is None or not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            raise ValueError("Input JSON is invalid or missing a 'nodes' list.")

        nodes: List[Dict[str, Any]] = data["nodes"]

        embedding_inputs: List[str] = []
        embedding_targets: List[Tuple[int, str]] = []

        for idx, node in enumerate(nodes):
            if isinstance(node, dict) and node.get("type") == "column":
                col_text = str(node.get("name", "") or "")
                tab_text = str(node.get("table", "") or "")

                node["column_embedding"] = []
                node["table_embedding"] = []

                if col_text:
                    embedding_inputs.append(col_text)
                    embedding_targets.append((idx, "column_embedding"))

                if tab_text:
                    embedding_inputs.append(tab_text)
                    embedding_targets.append((idx, "table_embedding"))

        if embedding_inputs:
            vectors = self.embedder.embed(embedding_inputs)
            if len(vectors) != len(embedding_inputs):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(embedding_inputs)}, got {len(vectors)}."
                )

            for vec, (node_idx, field_name) in zip(vectors, embedding_targets):
                nodes[node_idx][field_name] = vec

        FileWriter.write_json(data, output_path)
        return data


if __name__ == "__main__":
    appender = EntitiesEmbeddingAppender(model_name="text-embedding-3-small")
    input_file = "backend/src/query/output/extracted_graph_two_stage.json"
    output_file = "backend/src/query/output/extracted_graph_two_stage_embedding.json"

    try:
        updated_data = appender.process(input_file, output_file)
        print("Embedding process completed. Output saved to:", output_file)
        print("Updated nodes sample:", updated_data.get("nodes", [])[:2])
    except Exception as e:
        print("Error during processing:", str(e))