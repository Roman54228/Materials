# BPE Tokenization — Материалы к видео

Код и примеры к видео о том, как работает **BPE (Byte Pair Encoding)** — алгоритм токенизации, который используют GPT, Claude, LLaMA и другие LLM.

## Файлы

### Скрипты

| Файл | Описание |
|------|----------|
| `tiktoken_demo.py` | Демонстрация `tiktoken` — токенизатора OpenAI. Сравнение кодировок GPT-2/GPT-4/GPT-4o, разница в токенах между русским и английским, подводные камни (пробелы, регистр, числа), подсчёт стоимости API-запроса. Есть интерактивный режим. |
| `bpe_pretrained.py` | Исследование готового BPE-токенизатора из HuggingFace. Загружает любую модель (GPT-2, LLaMA, Mistral и др.), показывает первые/последние merge-правила, статистику словаря и пошаговую токенизацию. |
| `check_tokenizers.py` | Проверка типов токенизаторов популярных моделей (GPT-2, Qwen, Mistral, DeepSeek, Phi и др.) — определяет BPE/WordPiece/Unigram и размер словаря. |
| `tokenize_text.py` | BPE-токенизатор из готового файла `merges.txt`. Реализует полный пайплайн GPT-2: pre-tokenization → ByteLevel-кодирование → применение merge-правил. Поддерживает ввод из аргументов, файла или stdin. |

### Данные

| Файл | Описание |
|------|----------|
| `merges.txt` | Файл merge-правил BPE (используется в `tokenize_text.py`). |
| `vocab.json` | Словарь токенов (маппинг токен → ID). |

## Установка зависимостей

```bash
pip install tiktoken tokenizers transformers
```

## Быстрый старт

```bash
# Демонстрация tiktoken (GPT)
python tiktoken_demo.py

# Исследование токенизатора GPT-2
python bpe_pretrained.py openai-community/gpt2

# Проверка типов токенизаторов разных моделей
python check_tokenizers.py

# Токенизация текста из merges.txt
python tokenize_text.py "Hello, world!"
```
