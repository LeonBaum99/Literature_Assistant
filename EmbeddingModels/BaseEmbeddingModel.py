from abc import ABC, abstractmethod
import numpy as np


class BaseEmbeddingModel(ABC):
    """
    Abstract interface for embedding models to ensure compatibility
    across your testbed.
    """

    @abstractmethod
    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of texts into a numpy array of embeddings.
        Output shape: (num_texts, embedding_dimension)
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the size of the embedding vector (e.g., 768)."""
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model and tokenizer into memory (VRAM/RAM).
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Explicitly frees memory (VRAM/RAM) by deleting the model
        and clearing CUDA/MPS cache.
        """
        pass