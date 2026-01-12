import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import List
from openai import OpenAI
from backend.src.config.config import OPENAI_API_KEY

class EmbeddingClient:
    """
    A client wrapper for generating embeddings using OpenAI models.
    """

    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        """
        Initialize the Embedding client.

        Args:
            model_name (str): The embedding model to be used.
                              Defaults to 'text-embedding-3-large'.
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [d.embedding for d in response.data]


if __name__ == "__main__":
    ec = EmbeddingClient(model_name="text-embedding-3-small")  # or text-embedding-3-large
    vectors = ec.embed(["CustomerID", "transactions_1k"])
    print(f"Got {len(vectors)} embeddings, dim={len(vectors[0])}")
    print(vectors[0][:10])  # preview first 10 dims
