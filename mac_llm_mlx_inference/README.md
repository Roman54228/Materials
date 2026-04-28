# Как замерять скорость работы MLX и Ollama

Небольшой набор утилит для замера скорости генерации LLM на Mac:

- `mac.py` — локальный бенчмарк для моделей через `mlx_lm`.
- `ol.py` — бенчмарк для моделей через API Ollama.

## Что измеряется

Скрипты считают базовые метрики производительности:

- `ttft_s` — time to first token (время до первого токена).
- `wall_time_s` — общее время запроса.
- `prompt_tok_s` — скорость обработки prompt (токенов/сек).
- `gen_tok_s` — скорость генерации (токенов/сек).
- агрегаты по сериям запусков: min / mean / median / p95 / std.

## Требования

- macOS на Apple Silicon.
- Python 3.10+.
- Для `mac.py`: установленный `mlx-lm`.
- Для `ol.py`: локально запущенный Ollama.

## Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlx-lm requests
```

## Запуск бенчмарка MLX (`mac.py`)

Быстрый пример:

```bash
python mac.py \
  --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
  --prompt "Explain diffusion models shortly." \
  --num-predict 256 \
  --warmup 3 \
  --runs 10
```

Сохранить подробный отчет в JSON:

```bash
python mac.py --json-out mlx_bench.json
```

## Запуск бенчмарка Ollama (`ol.py`)

Пример:

```bash
python ol.py \
  --url http://localhost:11434/api/generate \
  --model qwen3:0.6b \
  --stream \
  --num-predict 256 \
  --warmup 3 \
  --runs 10
```

JSON-отчет:

```bash
python ol.py --stream --json-out ollama_bench.json
```

## Встроенный замер скорости через `mlx_lm.benchmark`

Кроме вашего скрипта `mac.py`, скорость можно измерять встроенными средствами `mlx-lm`:

```bash
python -m mlx_lm.benchmark --model mlx-community/Meta-Llama-3-8B-Instruct-4bit -p 512 -g 512
```

Где:

- `-p 512` — длина входного prompt (prefill tokens).
- `-g 512` — количество генерируемых токенов.

Так удобно быстро сравнивать разные модели и конфигурации на одной машине.

## Практика честного сравнения

- Прогоняйте несколько запусков (`warmup` + `runs`), а не один.
- Фиксируйте одинаковый prompt и параметры генерации.
- Сравнивайте cold-start и warm-start отдельно.
- Закрывайте тяжелые фоновые задачи перед бенчем.
