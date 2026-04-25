import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def run_data_command(args):
    from src.pipelines.factory import DataFactory

    factory = DataFactory()
    commands = {
        "full": factory.run_full_pipeline,
        "pdf-mine": factory.run_pdf_mining,
        "web-spider": factory.run_web_spider,
        "generate": factory.run_glossary_gen,
        "test-set": factory.create_test_set,
    }
    commands[args.data_command]()


def run_train_command(_args):
    from src.pipelines.factory import ModelFactory

    ModelFactory().train()


def run_benchmark_command(_args):
    from src.pipelines.factory import ModelFactory

    ModelFactory().run_benchmark()


def run_infer_command(args):
    from src.pipelines.factory import ModelFactory

    ModelFactory().translate(
        text=args.text,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        checkpoint_path=args.checkpoint,
    )


def run_maintenance_command(args):
    if args.maintenance_command == "clean-data":
        from scripts.clean_data import main

        main()
    elif args.maintenance_command == "download-docs":
        from scripts.download_extra_docs import download_files

        download_files()
    elif args.maintenance_command == "export-glossary":
        from scripts.export_glossary import main

        main()


def build_parser():
    parser = argparse.ArgumentParser(description="LEO project command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Run data pipeline tasks")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    for name in ("full", "pdf-mine", "web-spider", "generate", "test-set"):
        data_subparsers.add_parser(name)
    data_parser.set_defaults(func=run_data_command)

    train_parser = subparsers.add_parser("train", help="Train or resume the model")
    train_parser.set_defaults(func=run_train_command)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run model benchmark")
    benchmark_parser.set_defaults(func=run_benchmark_command)

    infer_parser = subparsers.add_parser("infer", help="Translate a single text")
    infer_parser.add_argument("--src-lang", default="eng_Latn", help="Source language code")
    infer_parser.add_argument("--tgt-lang", default="ita_Latn", help="Target language code")
    infer_parser.add_argument("--text", required=True, help="Text to translate")
    infer_parser.add_argument("--checkpoint", help="Optional checkpoint path")
    infer_parser.set_defaults(func=run_infer_command)

    maintenance_parser = subparsers.add_parser("maintenance", help="Run maintenance utilities")
    maintenance_subparsers = maintenance_parser.add_subparsers(dest="maintenance_command", required=True)
    for name in ("clean-data", "download-docs", "export-glossary"):
        maintenance_subparsers.add_parser(name)
    maintenance_parser.set_defaults(func=run_maintenance_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
