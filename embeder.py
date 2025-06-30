from FlagEmbedding import BGEM3FlagModel
import config

class EmbeddingAdapter:
    def __init__(self, model):
        self.model = model

    def embed(self, doc):
        return self.model.encode(doc)[config.DENSE_VECS]

def load_embedding_model():
    return BGEM3FlagModel(config.MODEL_PATH, use_fp16=config.USE_FP16)
