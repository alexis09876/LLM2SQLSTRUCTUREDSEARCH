import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import math
from typing import List, Tuple, Dict


class Utils:
    """
    A utility class for working with embedding vectors.
    - Supports cosine similarity between two vectors
    - Supports batch similarity search (query vs. many candidates)
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            vec1 (List[float]): First embedding vector
            vec2 (List[float]): Second embedding vector

        Returns:
            float: Cosine similarity in range [-1, 1]
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector lengths do not match: {len(vec1)} vs {len(vec2)}")

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    @staticmethod
    def rank_candidates(
        query_vec: List[float],
        candidates: Dict[str, List[float]],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate embeddings by cosine similarity to the query vector.

        Args:
            query_vec (List[float]): The embedding of the query
            candidates (Dict[str, List[float]]): Dict mapping candidate key -> embedding
            top_k (int): Number of top matches to return

        Returns:
            List[Tuple[str, float]]: Sorted list of (key, similarity) in descending order
        """
        scores = []
        for key, vec in candidates.items():
            try:
                sim = Utils.cosine_similarity(query_vec, vec)
                scores.append((key, sim))
            except ValueError:
                # Skip if dimension mismatch
                continue

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
