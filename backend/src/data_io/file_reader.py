import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
from typing import Any, Dict, List, Union, Optional

class FileReader:
    """
    A utility class for reading either JSON file or plain text prompt template.
    - Supports JSON files
    - Supports BIRD/Spider dev_tables.json (returns list[dict])
    """
        
    @staticmethod
    def load_json(file_path: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Loads a JSON file and returns the data as a dictionary.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: Parsed JSON data, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {file_path}.")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {file_path}. Invalid JSON format.")
        except Exception as e:
            print(f"Error: An error occurred while loading JSON file {file_path}: {e}")
        return None

    @staticmethod
    def load_dev_tables(path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Specifically load a dev_tables.json file.

        Args:
            path (str): Path to the dev_tables.json file.

        Returns:
            list[dict]: Parsed database schema entries if valid. 
                        Returns None if the file is invalid.
        """
        data = FileReader.load_json(path)
        if data is None:
            return None
        if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
            print(f"[Error] {path} is not a valid dev_tables.json (expect list[dict])")
            return None
        return data

    @staticmethod
    def read_prompt_template_from_txt(file_path: str) -> Optional[str]:
        """
        Reads a prompt template string from a text file.

        This function opens a text file from the given path, reads its content,
        and returns it as a string. It's useful for loading prompt template data from
        external files.

        Args:
            file_path (str): The path to the text file containing the prompt template.

        Returns:
            str: The content of the text file as a string, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt_template = file.read()
            return prompt_template
        except FileNotFoundError:
            print(f"File not found: {file_path}. Please check the file path.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
            return None

