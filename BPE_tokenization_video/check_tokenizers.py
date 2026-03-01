"""
Скрипт для проверки типов токенизаторов популярных LLM моделей.
Загружает токенизаторы через HuggingFace и определяет их тип (BPE, WordPiece, Unigram и т.д.).
"""

from transformers import AutoTokenizer
from tokenizers import Tokenizer


MODELS = {
    # GPT-семейство (OpenAI open-source)
    "GPT-2": "openai-community/gpt2",
    # Qwen
    "Qwen3": "Qwen/Qwen3-0.6B",
    # LLaMA
    
    # Mistral
    "Mistral-v0.3": "mistralai/Mistral-7B-v0.3",
    # Gemma
    # DeepSeek
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    # Phi
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "GPT-OSS": "openai/gpt-oss-120b",
    "MiniMax": "MiniMaxAI/MiniMax-M2.5"
}


def get_tokenizer_type(tokenizer) -> str:
    """Определяет тип токенизатора через backend_tokenizer (fast tokenizer)."""
    if not hasattr(tokenizer, "backend_tokenizer"):
        return "slow tokenizer (тип не определён автоматически)"

    backend = tokenizer.backend_tokenizer
    model = backend.model

    # Имя класса backend модели — самый надёжный способ
    class_name = type(model).__name__
    mapping = {
        "BPE": "BPE",
        "WordPiece": "WordPiece",
        "Unigram": "Unigram",
        "WordLevel": "WordLevel",
    }
    return mapping.get(class_name, f"Unknown ({class_name})")


def main():
    print(f"{'Модель':<20} {'Тип токенизатора':<20} {'Vocab size':<12} {'HF repo'}")
    print("-" * 90)

    for name, repo in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            tok_type = get_tokenizer_type(tok)
            vocab_size = tok.vocab_size
            print(f"{name:<20} {tok_type:<20} {vocab_size:<12} {repo}")
        except Exception as e:
            err = str(e).split("\n")[0][:60]
            print(f"{name:<20} {'ОШИБКА':<20} {'—':<12} {repo}  ({err})")


if __name__ == "__main__":
    main()
