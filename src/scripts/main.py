import argparse

from download_artifacts import download_artifacts
from export_classifier_to_onnx import export_classifier_to_onnx
from export_sentence_transformer_to_onnx import export_model_to_onnx
from settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["download-artifacts", "export-onnx"],
    )
    args = parser.parse_args()

    settings = Settings()

    if args.command == "download-artifacts":
        download_artifacts(settings)

    if args.command == "export-onnx":
        export_model_to_onnx(settings)
        export_classifier_to_onnx(settings)


if __name__ == "__main__":
    main()