from embeddingModels.ModernBertEmbedder import ModernBertEmbedder
from embeddingModels.QwenEmbedder import QwenEmbedder


class EmbeddingService:
    def __init__(self):
        self._models = {}

    def load_model(self, model_name: str):
        """Lazy loads the model only when requested."""
        if model_name in self._models:
            return self._models[model_name]

        print(f"Loading Embedding Model: {model_name}...")
        if model_name == "bert":
            self._models[model_name] = ModernBertEmbedder(
                model_name="Alibaba-NLP/gte-modernbert-base", normalize=True
            )
        elif model_name == "qwen":
            self._models[model_name] = QwenEmbedder("Qwen/Qwen3-Embedding-8B", use_fp16=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return self._models[model_name]

    def encode(self, text_list: list, model_name: str = "bert"):
        model = self.load_model(model_name)
        # TODO: Assuming your classes return numpy arrays
        return model.encode(text_list)