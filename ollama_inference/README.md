# Примеры запуска LLM из Python

Небольшие самостоятельные скрипты для работы с моделями четырьмя способами:

- через локальный [Ollama](https://ollama.com/);
- через локальный сервер LM Studio;
- напрямую с Hugging Face и `transformers`;
- через OpenAI API, в том числе с изображением.

## Что нужно заранее

- Python 3.11 или новее;
- [uv](https://docs.astral.sh/uv/) — менеджер Python-зависимостей;
- для локальных моделей: Ollama **или** LM Studio;
- для скриптов `openai_*.py`: ключ OpenAI API.

На macOS установить `uv` можно так:

```bash
brew install uv
```

Либо воспользуйтесь [официальной инструкцией по установке uv](https://docs.astral.sh/uv/getting-started/installation/).

## Установка проекта

Клонируйте репозиторий и установите все зависимости одной командой:

```bash
git clone https://github.com/<ваш-логин>/lm_query.git
cd lm_query
uv sync
```

`uv sync` создаст виртуальное окружение `.venv` и установит пакеты из `pyproject.toml`. Активировать окружение не требуется: далее используйте `uv run`.

> `torch` и `transformers` нужны только для `huggingface_v.py`, но устанавливаются вместе с остальными зависимостями для простоты первого запуска.

## Ollama

1. Установите Ollama с [ollama.com](https://ollama.com/).
2. Скачайте модель, указанную в примере:

   ```bash
   ollama pull qwen3:0.6b
   ```

3. Убедитесь, что сервер Ollama запущен. Обычно приложение запускает его само; при необходимости выполните в отдельном терминале:

   ```bash
   ollama serve
   ```

4. Запустите пример:

   ```bash
   uv run python ollama_v.py
   ```

Список установленных моделей: `ollama list`. Для другой модели поменяйте значение `model` в `ollama_v.py` и заранее выполните `ollama pull <имя-модели>`.

## LM Studio

1. Установите [LM Studio](https://lmstudio.ai/), скачайте и загрузите любую чат-модель.
2. На вкладке **Developer** запустите локальный сервер. В примере ожидается адрес `http://127.0.0.1:1234`.
3. Укажите в `lmstudio.py` точный идентификатор загруженной модели — он должен совпадать с именем, которое LM Studio показывает в сервере.
4. Выполните:

   ```bash
   uv run python lmstudio.py
   ```

## Hugging Face без отдельного сервера

Этот вариант сам скачает `Qwen/Qwen3-0.6B` при первом запуске. Файлы модели попадут в кэш Hugging Face; при следующих запусках загрузка не потребуется.

```bash
uv run python huggingface_v.py
```

Скрипт выберет ускорение автоматически: CUDA на NVIDIA, MPS на Apple Silicon или CPU в остальных случаях. Для другой модели замените `MODEL_ID` в файле.

## OpenAI API

Создайте рядом со скриптами файл `.env`:

```dotenv
OPENAI_API_KEY=sk-...
```

`.env` уже исключён из Git, поэтому ключ не попадёт в репозиторий. Затем запустите текстовый или мультимодальный пример:

```bash
uv run python openai_v.py
uv run python openai_image.py
```

`openai_image.py` читает приложенный к репозиторию файл `pasp.webp`. Чтобы проверить другое изображение, замените путь в константе `IMAGE_PATH`.

## Состав репозитория

| Файл | Что демонстрирует |
| --- | --- |
| `ollama_v.py` | запрос к локальному Ollama через совместимый с OpenAI API клиент |
| `lmstudio.py` | запрос к серверу LM Studio |
| `huggingface_v.py` | загрузку и инференс модели через `transformers` |
| `openai_v.py` | текстовый запрос к OpenAI Responses API |
| `openai_image.py` | отправку изображения в OpenAI Responses API |

## Частые проблемы

- **`Connection refused` в Ollama или LM Studio** — запустите соответствующий локальный сервер и проверьте порт в скрипте.
- **`model not found` в Ollama** — выполните `ollama pull` для модели, указанной в `model`.
- **Ошибка `OPENAI_API_KEY`** — создайте `.env` по примеру выше или задайте переменную окружения в терминале.
- **Hugging Face работает медленно** — модель может выполняться на CPU. На Mac убедитесь, что установлена актуальная версия PyTorch с поддержкой MPS.
