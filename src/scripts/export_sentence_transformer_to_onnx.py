import os

import torch
from transformers import AutoModel, AutoTokenizer

from settings import Settings


class SentenceEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled


def export_model_to_onnx(settings: Settings):
    base_model = AutoModel.from_pretrained(settings.sentence_transformer_dir)
    tokenizer = AutoTokenizer.from_pretrained(settings.sentence_transformer_dir)

    model = SentenceEmbeddingModel(base_model)
    model.eval()

    dummy_text = "This is a sample input for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    onnx_path = settings.onnx_embedding_model_path
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "sentence_embedding": {0: "batch_size"},
            },
            opset_version=18,
            dynamo=False,
        )

    settings.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(settings.tokenizer_dir)

    print(f"ONNX embedding model exported to {onnx_path}")
    print(f"Tokenizer exported to {settings.tokenizer_dir}")

    return onnx_path


if __name__ == "__main__":
    export_model_to_onnx(Settings())