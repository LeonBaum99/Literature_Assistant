from backend.config import settings  # Import settings
from embeddingModels.ModernBertEmbedder import ModernBertEmbedder
from embeddingModels.QwenEmbedder import QwenEmbedder


class EmbeddingService:
    def __init__(self):
        self._models = {}

    def load_model(self, model_key: str):
        if model_key in self._models:
            return self._models[model_key]

        print(f"Loading Model Key: {model_key}...")

        # USE CONFIG HERE
        if model_key == "bert":
            self._models[model_key] = ModernBertEmbedder(
                model_name=settings.models.bert,
                normalize=True
            )
        elif model_key == "qwen":
            self._models[model_key] = QwenEmbedder(
                model_name=settings.models.qwen,
                use_fp16=True
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        return self._models[model_key]

    def encode(self, text_list: list, model_name: str = "bert"):
        model = self.load_model(model_name)
        return model.encode(text_list)
