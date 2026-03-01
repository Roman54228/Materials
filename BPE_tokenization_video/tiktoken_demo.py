"""
╔══════════════════════════════════════════════════════════════╗
║          tiktoken — Токенизатор OpenAI на практике           ║
║        Смотрим как GPT разбивает текст на токены             ║
╚══════════════════════════════════════════════════════════════╝

pip install tiktoken
"""

import tiktoken
import time

# ─────────────────────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────────────────────
PAUSE = 0.3

class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    RESET   = "\033[0m"
    BG_DARK = "\033[48;5;236m"

# Палитра фонов для визуализации токенов (чередуем)
TOKEN_COLORS = [
    "\033[48;5;22m",   # тёмно-зелёный
    "\033[48;5;24m",   # тёмно-синий
    "\033[48;5;88m",   # тёмно-красный
    "\033[48;5;58m",   # оливковый
    "\033[48;5;54m",   # фиолетовый
    "\033[48;5;94m",   # коричневый
    "\033[48;5;30m",   # бирюзовый
    "\033[48;5;52m",   # бордовый
]

def header(title):
    w = 62
    print()
    print(f"{C.CYAN}{'═' * w}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{C.CYAN}{'═' * w}{C.RESET}")
    print()

def pause():
    time.sleep(PAUSE)

def colorize_tokens(text, enc):
    """Раскрасить каждый токен в тексте своим цветом."""
    tokens = enc.encode(text)
    result = ""
    for i, tok in enumerate(tokens):
        decoded = enc.decode([tok])
        bg = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        result += f"{bg}{C.BOLD} {decoded} {C.RESET}"
    return result


# ═════════════════════════════════════════════════════════════
#  ШАГ 1: Доступные кодировки (модели)
# ═════════════════════════════════════════════════════════════

header("ШАГ 1: Кодировки tiktoken — какая модель что использует")

encodings = [
    ("gpt2",         "gpt-2",              "r50k_base",   50257),
    ("gpt-3.5-turbo","GPT-3.5 Turbo",      "cl100k_base", 100256),
    ("gpt-4",        "GPT-4",              "cl100k_base", 100256),
    ("gpt-4o",       "GPT-4o",             "o200k_base",  200019),
]

print(f"  {'Модель':<18} {'Кодировка':<15} {'Размер словаря':>15}")
print(f"  {C.DIM}{'─' * 50}{C.RESET}")

for model, name, enc_name, vocab_size in encodings:
    print(f"  {C.GREEN}{name:<18}{C.RESET} {C.CYAN}{enc_name:<15}{C.RESET} {C.YELLOW}{vocab_size:>15,}{C.RESET}")

print()
print(f"  {C.DIM}GPT-2 знал ~50k токенов, GPT-4o — уже ~200k!{C.RESET}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 2: Базовая токенизация
# ═════════════════════════════════════════════════════════════

header("ШАГ 2: Базовая токенизация — encode / decode")

enc = tiktoken.get_encoding("cl100k_base")
text = "Hello, world! This is tokenization."

tokens = enc.encode(text)

print(f'  Текст:   {C.GREEN}"{text}"{C.RESET}')
print(f'  Кодировка: {C.CYAN}cl100k_base{C.RESET} (GPT-4)')
print()
print(f"  {C.BOLD}Токен IDs:{C.RESET}  {C.YELLOW}{tokens}{C.RESET}")
print(f"  {C.BOLD}Кол-во токенов:{C.RESET} {C.RED}{len(tokens)}{C.RESET}")
print()

print(f"  {C.BOLD}Разбивка по токенам:{C.RESET}")
print()
for i, tok in enumerate(tokens):
    decoded = enc.decode([tok])
    byte_repr = decoded.encode("utf-8")
    hex_bytes = " ".join(f"{b:02X}" for b in byte_repr)
    display = repr(decoded)
    print(f"    {C.YELLOW}{i:3d}{C.RESET}  │  ID: {C.CYAN}{tok:>6d}{C.RESET}"
          f"  │  {C.GREEN}{display:<20}{C.RESET}"
          f"  │  bytes: [{C.DIM}{hex_bytes}{C.RESET}]")

print()
print(f"  {C.BOLD}Визуализация:{C.RESET}")
print(f"  {colorize_tokens(text, enc)}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 3: Русский текст — больше токенов!
# ═════════════════════════════════════════════════════════════

header("ШАГ 3: Русский vs Английский — разница в токенах")

pairs = [
    ("Hello world",        "Привет мир"),
    ("artificial intelligence", "искусственный интеллект"),
    ("The cat sat on the mat", "Кот сидел на коврике"),
    ("tokenization",       "токенизация"),
]

print(f"  {'Текст':<30} {'Язык':>6}  {'Токены':>7}  {'Символов':>9}  {'Ток/сим':>8}")
print(f"  {C.DIM}{'─' * 68}{C.RESET}")

for en, ru in pairs:
    en_tokens = enc.encode(en)
    ru_tokens = enc.encode(ru)

    en_ratio = len(en_tokens) / len(en)
    ru_ratio = len(ru_tokens) / len(ru)

    print(f"  {C.GREEN}{en:<30}{C.RESET} {'EN':>6}  {C.YELLOW}{len(en_tokens):>7}{C.RESET}"
          f"  {len(en):>9}  {C.DIM}{en_ratio:>8.2f}{C.RESET}")
    print(f"  {C.MAGENTA}{ru:<30}{C.RESET} {'RU':>6}  {C.RED}{len(ru_tokens):>7}{C.RESET}"
          f"  {len(ru):>9}  {C.DIM}{ru_ratio:>8.2f}{C.RESET}")
    print()

print(f"  {C.DIM}Кириллица — дороже! Один русский символ ≈ 2-3 токена,{C.RESET}")
print(f"  {C.DIM}потому что BPE обучался в основном на английском тексте.{C.RESET}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 4: Что «видит» GPT — посимвольная раскраска
# ═════════════════════════════════════════════════════════════

header("ШАГ 4: Что видит GPT — визуализация токенов")

examples = [
    "The quick brown fox jumps over the lazy dog",
    "Привет! Как дела? Всё отлично!",
    "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "🎉🚀💡 Emojis are tokens too!",
    "GPT-4o is 10x cheaper than GPT-4-turbo!!!",
    "https://www.example.com/api/v2/users?page=1&limit=50",
]

for text in examples:
    tokens = enc.encode(text)
    print(f"  {C.DIM}[{len(tokens)} tokens]{C.RESET}")
    print(f"  {colorize_tokens(text, enc)}")
    print()

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 5: Сравнение кодировок (GPT-2 vs GPT-4 vs GPT-4o)
# ═════════════════════════════════════════════════════════════

header("ШАГ 5: Эволюция кодировок — GPT-2 → GPT-4 → GPT-4o")

enc_gpt2  = tiktoken.get_encoding("r50k_base")
enc_gpt4  = tiktoken.get_encoding("cl100k_base")
enc_gpt4o = tiktoken.get_encoding("o200k_base")

test_texts = [
    "Hello, how are you?",
    "Машинное обучение — это будущее",
    "def hello():\n    print('world')",
    "2024-01-15T14:30:00Z",
    "    indentation matters in Python",
]

print(f"  {'Текст':<40} {'GPT-2':>6} {'GPT-4':>6} {'GPT-4o':>7}")
print(f"  {C.DIM}{'─' * 62}{C.RESET}")

for text in test_texts:
    t2 = len(enc_gpt2.encode(text))
    t4 = len(enc_gpt4.encode(text))
    t4o = len(enc_gpt4o.encode(text))

    display = text.replace("\n", "\\n")
    if len(display) > 38:
        display = display[:35] + "..."

    # Подсветим лучший (минимальный) результат
    vals = [t2, t4, t4o]
    best = min(vals)

    def fmt(v):
        if v == best:
            return f"{C.GREEN}{C.BOLD}{v:>6}{C.RESET}"
        return f"{C.YELLOW}{v:>6}{C.RESET}"

    print(f"  {C.CYAN}{display:<40}{C.RESET} {fmt(t2)} {fmt(t4)} {fmt(t4o):>7}")

print()
print(f"  {C.GREEN}■{C.RESET} = лучший результат (меньше токенов)")
print(f"  {C.DIM}Новые кодировки эффективнее: больше словарь → меньше токенов{C.RESET}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 6: Необычные кейсы и ловушки
# ═════════════════════════════════════════════════════════════

header("ШАГ 6: Подводные камни токенизации")

enc = tiktoken.get_encoding("cl100k_base")

print(f"  {C.BOLD}1. Пробелы имеют значение:{C.RESET}")
print()
for text in ["dog", " dog", "  dog", "   dog"]:
    tokens = enc.encode(text)
    display = repr(text)
    print(f"    {C.CYAN}{display:<12}{C.RESET} → {C.YELLOW}{len(tokens)} ток.{C.RESET}  IDs: {tokens}")
print()

print(f"  {C.BOLD}2. Регистр меняет токенизацию:{C.RESET}")
print()
for text in ["hello", "Hello", "HELLO", "hElLo"]:
    tokens = enc.encode(text)
    print(f"    {C.CYAN}{text:<12}{C.RESET} → {C.YELLOW}{len(tokens)} ток.{C.RESET}  IDs: {tokens}")
print()

print(f"  {C.BOLD}3. Числа — дорогие!{C.RESET}")
print()
for text in ["100", "1000", "123456789", "3.14159265358979"]:
    tokens = enc.encode(text)
    print(f"    {C.CYAN}{text:<20}{C.RESET} → {C.YELLOW}{len(tokens)} ток.{C.RESET}  "
          f"{colorize_tokens(text, enc)}")
print()

print(f"  {C.BOLD}4. Повторяющийся текст:{C.RESET}")
print()
for n in [1, 3, 5, 10]:
    text = "ha" * n
    tokens = enc.encode(text)
    print(f"    {C.CYAN}{'ha' + f' ×{n}':<12}{C.RESET} → {C.YELLOW}{len(tokens)} ток.{C.RESET}  "
          f"{colorize_tokens(text, enc)}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 7: Подсчёт стоимости API-запроса
# ═════════════════════════════════════════════════════════════

header("ШАГ 7: Считаем стоимость API-запроса")

prompt = """You are a helpful assistant that translates Russian to English.
Translate the following text:
Искусственный интеллект меняет мир технологий.
Нейронные сети обучаются на огромных объёмах данных."""

enc4o = tiktoken.get_encoding("o200k_base")
tokens = enc4o.encode(prompt)

# Примерные цены GPT-4o ($/1M tokens)
input_price = 2.50   # $/1M input tokens
output_price = 10.00  # $/1M output tokens

est_output_tokens = 30  # примерная оценка ответа
input_cost = len(tokens) * input_price / 1_000_000
output_cost = est_output_tokens * output_price / 1_000_000
total_cost = input_cost + output_cost

print(f"  {C.BOLD}Промпт:{C.RESET}")
for line in prompt.split("\n"):
    print(f"    {C.DIM}{line}{C.RESET}")
print()
print(f"  {C.BOLD}Кодировка:{C.RESET} {C.CYAN}o200k_base{C.RESET} (GPT-4o)")
print(f"  {C.BOLD}Входные токены:{C.RESET}  {C.YELLOW}{len(tokens)}{C.RESET}")
print(f"  {C.BOLD}≈ Выходные токены:{C.RESET} {C.YELLOW}~{est_output_tokens}{C.RESET}")
print()
print(f"  {C.BOLD}Стоимость (GPT-4o):{C.RESET}")
print(f"    Input:  {len(tokens):>5} × ${input_price}/1M = {C.GREEN}${input_cost:.6f}{C.RESET}")
print(f"    Output: {est_output_tokens:>5} × ${output_price}/1M = {C.GREEN}${output_cost:.6f}{C.RESET}")
print(f"    {C.DIM}{'─' * 40}{C.RESET}")
print(f"    {C.BOLD}Итого: {C.GREEN}${total_cost:.6f}{C.RESET}")
print()
print(f"  {C.DIM}Вывод: один запрос стоит доли цента.{C.RESET}")
print(f"  {C.DIM}Но на миллионах запросов — токены решают!{C.RESET}")

pause()


# ═════════════════════════════════════════════════════════════
#  ШАГ 8: Интерактивный режим
# ═════════════════════════════════════════════════════════════

header("ШАГ 8: Попробуй сам! (Ctrl+C чтобы выйти)")

# enc = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.encoding_for_model("gpt-4")
try:
    while True:
        print(f"  {C.BOLD}Введи текст:{C.RESET} ", end="")
        text = input()
        if not text:
            continue

        tokens = enc.encode(text)
        print()
        print(f"  Токенов: {C.YELLOW}{C.BOLD}{len(tokens)}{C.RESET}")
        print(f"  Визуализация:")
        print(f"  {colorize_tokens(text, enc)}")
        print()
        print(f"  Разбивка:")
        for i, tok in enumerate(tokens):
            decoded = enc.decode([tok])
            print(f"    {C.YELLOW}{i:3d}{C.RESET}  ID: {C.CYAN}{tok:>6d}{C.RESET}"
                  f"  → {C.GREEN}{repr(decoded)}{C.RESET}")
        print()
except KeyboardInterrupt:
    print()
    print()
    print(f"{C.CYAN}{'═' * 62}")
    print(f"  {C.BOLD}Спасибо за просмотр! Подписывайтесь 🔥{C.RESET}")
    print(f"{C.CYAN}{'═' * 62}{C.RESET}")
    print()
