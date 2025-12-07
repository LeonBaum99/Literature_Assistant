import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Union, List, Any
from tqdm import tqdm
import gc

from EmbeddingModels.BaseEmbeddingModel import BaseEmbeddingModel


class ModernBertEmbedder(BaseEmbeddingModel):

    def __init__(
            self,
            model_name: str = "answerdotai/ModernBERT-base",
            device: str = None,
            normalize: bool = True,
            auto_load: bool = True
    ):
        self.model_name = model_name
        self.normalize = normalize

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.model = None
        self.tokenizer = None

        if auto_load:
            self.load()

    def load(self) -> None:
        if self.model is not None:
            print(f"{self.model_name} is already loaded.")
            return

        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def unload(self) -> None:
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

    def get_underlying_model(self) -> Any:
        if self.model is None:
            self.load()
        return self.model

    @property
    def dimension(self) -> int:
        if self.model is None:
            self.load()
        return self.model.config.hidden_size

    def _mean_pooling(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        # Auto-load check
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i: i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])

                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                del inputs, outputs, embeddings

        if not all_embeddings:
            return np.array([])

        return np.vstack(all_embeddings)