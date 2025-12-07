from typing import Union, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import gc

from EmbeddingModels.BaseEmbeddingModel import BaseEmbeddingModel


class QwenEmbedder(BaseEmbeddingModel):
    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-Embedding-8B",
            device: str = None,
            normalize: bool = True,
            use_fp16: bool = True,
            auto_load: bool = True
    ):
        """
        Args:
            auto_load: If True, loads the model immediately on initialization.
        """
        self.model_name = model_name
        self.normalize = normalize
        self.use_fp16 = use_fp16  # Stored for use in load()

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None

        if auto_load:
            self.load()

    def load(self) -> None:
        """Loads the model and tokenizer if not already loaded."""
        if self.model is not None:
            print(f"{self.model_name} is already loaded.")
            return

        print(f"Loading {self.model_name} on {self.device}...")

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        ).to(self.device)

        self.model.eval()

    def unload(self) -> None:
        """Unloads model to free VRAM."""
        print(f"Unloading {self.model_name}...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

        print("Model unloaded.")

    @property
    def dimension(self) -> int:
        # Check if model is loaded, otherwise we might need to load config separately
        # For simplicity, we assume model is loaded or we load it temporarily.
        if self.model is None:
            # Fallback: load just config or force load.
            # Usually safe to assume user calls this when model is ready.
            self.load()
        return self.model.config.hidden_size

    def _last_token_pooling(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return last_hidden_states[:, -1, :]

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 4,
               instruction: str = "",
               show_progress: bool = False) -> np.ndarray:

        # Auto-load if the user forgot
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        if instruction:
            texts = [f"{instruction}{t}" for t in texts]

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Encoding (Qwen)", unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i: i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = self._last_token_pooling(outputs.last_hidden_state, inputs['attention_mask'])

                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

                # Cleanup batch VRAM
                del inputs, outputs, embeddings

        if not all_embeddings:
            return np.array([])

        return np.vstack(all_embeddings)