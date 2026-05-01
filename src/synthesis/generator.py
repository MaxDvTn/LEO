# src/synthesis/generator.py
from src.synthesis.base import BaseGenerator
from src.common.config import conf


_CLOUD_PREFIXES = ("openai/", "anthropic/", "google/", "deepseek/")


def get_generator(model_id: str = None) -> BaseGenerator:
    mid = model_id or conf.gen.model_id

    if mid.startswith("ollama/"):
        from src.synthesis.ollama_generator import OllamaGenerator
        return OllamaGenerator(model_id=mid)

    if mid.startswith(_CLOUD_PREFIXES):
        from src.synthesis.cloud_generator import CloudGenerator
        return CloudGenerator(model_id=mid)

    raise ValueError(
        f"Unsupported model_id '{mid}'. "
        "Use 'ollama/<model>' for local inference or "
        "'openai/<model>', 'anthropic/<model>', 'google/<model>', 'deepseek/<model>' for cloud."
    )


SyntheticGenerator = get_generator


if __name__ == "__main__":
    gen = get_generator()
    df = gen.generate_dataset(terms=["guarnizione termoacustica"])
    print(df.head())