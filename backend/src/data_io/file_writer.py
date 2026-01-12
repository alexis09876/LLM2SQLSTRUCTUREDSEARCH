import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Any

class FileWriter:
    """
    A utility class for writing data to JSON file.
    """

    @staticmethod
    def ensure_dir(path: str) -> None:
        """
        Ensure the directory for the given file path exists.

        Args:
            path (str): The file path for which the directory should be ensured.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

    @staticmethod
    def write_json(data: Any, path: str, indent: int = 2) -> None:
        """
        Write data to a JSON file.

        Args:
            data (Any): The data to be serialized into JSON.
            path (str): The path where the JSON file will be saved.
            indent (int, optional): Number of spaces for indentation. Defaults to 2.

        Returns:
            None
        """
        FileWriter.ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"[INFO] JSON written to {path}")
