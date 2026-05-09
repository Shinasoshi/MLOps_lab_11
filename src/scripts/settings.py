from pathlib import Path


class Settings:
    aws_region = "us-east-1"
    s3_bucket = "mlops-lab11-models"

    project_root = Path(__file__).resolve().parents[2]

    model_dir = project_root / "model"

    sentence_transformer_dir = model_dir / "sentence_transformer.model"
    classifier_joblib_path = model_dir / "classifier.joblib"

    onnx_dir = model_dir / "onnx"
    onnx_embedding_model_path = onnx_dir / "embedding_model.onnx"
    onnx_classifier_path = onnx_dir / "classifier.onnx"

    tokenizer_dir = onnx_dir / "tokenizer"
    onnx_tokenizer_path = tokenizer_dir / "tokenizer.json"

    embedding_dim = 384