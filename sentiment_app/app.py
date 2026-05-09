from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from tokenizers import Tokenizer


SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class PredictRequest(BaseModel):
    text: str


class SentimentPredictor:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]

        model_dir = project_root / "model" / "onnx"

        tokenizer_path = model_dir / "tokenizer" / "tokenizer.json"
        embedding_model_path = model_dir / "embedding_model.onnx"
        classifier_model_path = model_dir / "classifier.onnx"

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self.embedding_session = ort.InferenceSession(str(embedding_model_path))
        self.classifier_session = ort.InferenceSession(str(classifier_model_path))

    def predict(self, text: str) -> str:
        cleaned_text = text.strip()

        encoded = self.tokenizer.encode(cleaned_text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        embedding_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        embeddings = self.embedding_session.run(None, embedding_inputs)[0]

        classifier_input_name = self.classifier_session.get_inputs()[0].name

        classifier_inputs = {
            classifier_input_name: embeddings.astype(np.float32),
        }

        prediction = self.classifier_session.run(None, classifier_inputs)[0]

        label_id = int(prediction[0])

        return SENTIMENT_MAP.get(label_id, "unknown")


app = FastAPI()

predictor: SentimentPredictor | None = None


def get_predictor() -> SentimentPredictor:
    global predictor

    if predictor is None:
        predictor = SentimentPredictor()

    return predictor


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    model = get_predictor()
    label = model.predict(request.text)

    return {
        "text": request.text,
        "label": label,
    }


handler = Mangum(app)