"""
Исследуем готовый BPE-токенизатор с HuggingFace
=================================================
Загружаем токенизатор реальной модели и смотрим:
  1) Его merge-правила (первые, последние, самые интересные)
  2) Как он пошагово токенизирует текст
  3) Статистику словаря

Установка:  pip install tokenizers transformers
"""

import sys
import json
import tempfile
import os

# ─── Цвета ───────────────────────────────────────────────
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

def header(title):
    print(f"\n{C.CYAN}{'═' * 70}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{C.CYAN}{'═' * 70}{C.RESET}\n")

def subheader(title):
    print(f"  {C.YELLOW}── {title} ──{C.RESET}\n")

# ─── Декодирование ByteLevel токенов ─────────────────────
def _build_byte_decoder():
    """Строим обратный маппинг: Unicode-символ → байт (как в GPT-2 ByteLevel BPE)."""
    # Это стандартный bytes_to_unicode маппинг из GPT-2
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): bytes([b]) for b, c in zip(bs, cs)}

_BYTE_DECODER = _build_byte_decoder()

def decode_token_raw(t):
    """Декодирует ByteLevel-токен в реальные байты и затем в UTF-8 строку."""
    try:
        raw_bytes = b"".join(_BYTE_DECODER.get(c, c.encode("utf-8")) for c in t)
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return t

def decode_token(t):
    """Превращает ByteLevel-токен в читаемый вид для отображения."""
    raw = decode_token_raw(t)
    # Заменяем невидимые символы на видимые маркеры
    return raw.replace(" ", "·").replace("\n", "↵").replace("\t", "→")

# ─── Загрузка токенизатора ───────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "openai-community/gpt2"

header(f"Загрузка токенизатора: {MODEL_NAME}")

from tokenizers import Tokenizer

# Пробуем загрузить через transformers (поддерживает больше форматов)
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Получаем underlying fast tokenizer
if not hf_tokenizer.is_fast:
    print(f"  {C.RED}Этот токенизатор не является 'fast' (rust-based).{C.RESET}")
    print(f"  {C.RED}Merge-правила недоступны.{C.RESET}")
    sys.exit(1)

fast_tokenizer = hf_tokenizer.backend_tokenizer

# Извлекаем merge-правила из модели
# Сохраняем во временный файл и читаем merges.txt
tmp_dir = tempfile.mkdtemp()
try:
    fast_tokenizer.model.save(tmp_dir)

    merges_path = os.path.join(tmp_dir, "merges.txt")
    vocab_path = os.path.join(tmp_dir, "vocab.json")

    merges = []
    if os.path.exists(merges_path):
        with open(merges_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

    vocab = {}
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
finally:
    for fname in os.listdir(tmp_dir):
        os.unlink(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)

if not merges:
    print(f"  {C.RED}Не удалось извлечь merge-правила. Возможно, это не BPE-токенизатор.{C.RESET}")
    sys.exit(1)

vocab_by_id = {v: k for k, v in vocab.items()}

print(f"  {C.BOLD}Модель:{C.RESET}          {C.GREEN}{MODEL_NAME}{C.RESET}")
print(f"  {C.BOLD}Размер словаря:{C.RESET}  {C.GREEN}{len(vocab)}{C.RESET} токенов")
print(f"  {C.BOLD}Merge-правил:{C.RESET}    {C.GREEN}{len(merges)}{C.RESET}")
print(f"  {C.BOLD}Тип:{C.RESET}             {C.GREEN}BPE (Byte-Pair Encoding){C.RESET}")

# ═══════════════════════════════════════════════════════════
#  1. Первые merge-правила (самые частые пары)
# ═══════════════════════════════════════════════════════════
header("1. Первые merge-правила (самые частые пары в обучающих данных)")

print(f"  Эти пары символов встречались чаще всего в обучающем корпусе.")
print(f"  Они были объединены первыми при обучении BPE.\n")

SHOW_FIRST = 30
for i, (a, b) in enumerate(merges[:SHOW_FIRST]):
    merged = a + b
    a_d = decode_token(a)
    b_d = decode_token(b)
    m_d = decode_token(merged)
    # Покажем также ID результирующего токена
    token_id = vocab.get(merged, "?")
    print(f"    {C.YELLOW}{i+1:5d}.{C.RESET}  "
          f"{C.CYAN}{a_d!r:>12}{C.RESET} + {C.CYAN}{b_d!r:<12}{C.RESET}"
          f" → {C.GREEN}{C.BOLD}{m_d!r:<14}{C.RESET}"
          f" {C.DIM}(id={token_id}){C.RESET}")

# ═══════════════════════════════════════════════════════════
#  2. Последние merge-правила (самые редкие)
# ═══════════════════════════════════════════════════════════
header("2. Последние merge-правила (редкие / длинные токены)")

print(f"  Последние правила создают длинные токены из уже объединённых частей.")
print(f"  Это редкие, но целые слова или фразы.\n")

SHOW_LAST = 20
start = max(0, len(merges) - SHOW_LAST)
for i in range(start, len(merges)):
    a, b = merges[i]
    merged = a + b
    a_d = decode_token(a)
    b_d = decode_token(b)
    m_d = decode_token(merged)
    token_id = vocab.get(merged, "?")
    print(f"    {C.YELLOW}{i+1:5d}.{C.RESET}  "
          f"{C.CYAN}{a_d!r:>16}{C.RESET} + {C.CYAN}{b_d!r:<16}{C.RESET}"
          f" → {C.GREEN}{C.BOLD}{m_d!r:<20}{C.RESET}"
          f" {C.DIM}(id={token_id}){C.RESET}")

# ═══════════════════════════════════════════════════════════
#  3. Интересные паттерны в merge-правилах
# ═══════════════════════════════════════════════════════════
header("3. Статистика merge-правил")

# Подсчитаем длины результирующих токенов
result_lengths = [len(decode_token_raw(a + b)) for a, b in merges]
max_len = max(result_lengths)
avg_len = sum(result_lengths) / len(result_lengths)

print(f"  {C.BOLD}Средняя длина результата merge:{C.RESET}  {C.GREEN}{avg_len:.1f}{C.RESET} символов")
print(f"  {C.BOLD}Максимальная длина:{C.RESET}              {C.GREEN}{max_len}{C.RESET} символов")

# Самые длинные токены в словаре
print(f"\n  {C.BOLD}Самые длинные токены в словаре:{C.RESET}\n")
long_tokens = sorted(vocab.keys(), key=lambda t: len(decode_token_raw(t)), reverse=True)
for t in long_tokens[:15]:
    raw = decode_token_raw(t)
    display = decode_token(t)
    tid = vocab[t]
    print(f"    {C.GREEN}{display!r:<30}{C.RESET}  "
          f"({len(raw):2d} символов, id={C.DIM}{tid}{C.RESET})")

# Токены, начинающиеся с пробела vs без
space_tokens = [t for t in vocab if t.startswith("Ġ")]
no_space = [t for t in vocab if not t.startswith("Ġ") and t.isalpha()]
print(f"\n  {C.BOLD}Токены с пробелом в начале (·word):{C.RESET}  {C.GREEN}{len(space_tokens)}{C.RESET}")
print(f"  {C.BOLD}Токены без пробела (word):{C.RESET}          {C.GREEN}{len(no_space)}{C.RESET}")
print(f"  {C.DIM}(В ByteLevel BPE пробел кодируется как часть следующего слова){C.RESET}")

# ═══════════════════════════════════════════════════════════
#  4. Пошаговая токенизация текста
# ═══════════════════════════════════════════════════════════
header("4. Пошаговая токенизация — воспроизводим merge-и")

TEST_TEXTS = [
    "Hello world",
    "The quick brown fox",
    "tokenization",
    "unbelievable",
    "GPT-4 is amazing!",
]

for text in TEST_TEXTS:
    print(f'  Текст: {C.GREEN}"{text}"{C.RESET}')

    # Pre-tokenize как делает модель
    pre_tokenized = fast_tokenizer.pre_tokenizer.pre_tokenize_str(text)

    for word_str, _offsets in pre_tokenized:
        tokens = list(word_str)
        word_display = decode_token(word_str)
        print(f"    Слово: {C.MAGENTA}{word_display!r}{C.RESET}")
        print(f"      начало:    [{', '.join(C.DIM + repr(decode_token(t)) + C.RESET for t in tokens)}]")

        # Применяем merge-правила
        step = 0
        for merge_idx, (a, b) in enumerate(merges):
            i = 0
            changed = False
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    changed = True
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if changed:
                tokens = new_tokens
                step += 1
                a_d = decode_token(a)
                b_d = decode_token(b)
                m_d = decode_token(a + b)
                tok_display = ", ".join(
                    f"{C.GREEN}{decode_token(t)!r}{C.RESET}" for t in tokens
                )
                print(f"      merge {C.YELLOW}#{merge_idx+1:>5}{C.RESET}: "
                      f"{a_d!r}+{b_d!r} → {C.BOLD}{m_d!r}{C.RESET}  "
                      f"=> [{tok_display}]")

        print()

    # Финальный результат
    encoding = hf_tokenizer(text, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    token_strs = hf_tokenizer.convert_ids_to_tokens(token_ids)

    print(f"  {C.BOLD}Итоговые токены:{C.RESET}  ", end="")
    print(" ".join(f"{C.GREEN}{decode_token(t)!r}{C.RESET}" for t in token_strs))
    print(f"  {C.BOLD}Token IDs:{C.RESET}       {token_ids}")
    print(f"  {C.BOLD}Декодировано:{C.RESET}    {C.CYAN}{hf_tokenizer.decode(token_ids)!r}{C.RESET}")
    print(f"\n  {'─' * 60}\n")


# ═══════════════════════════════════════════════════════════
#  5. Сравнение: одно слово — разные стадии merge-ов
# ═══════════════════════════════════════════════════════════
header("5. Эволюция токенизации слова по мере применения merge-правил")

DEMO_TEXT = "unbelievable"
print(f'  Текст: {C.GREEN}"{DEMO_TEXT}"{C.RESET}\n')

pre_tokenized = fast_tokenizer.pre_tokenizer.pre_tokenize_str(DEMO_TEXT)

checkpoints = [0, 10, 50, 100, 500, 1000, 5000, 10000, 20000, len(merges)]
checkpoints = sorted(set(c for c in checkpoints if c <= len(merges)))

for num_merges in checkpoints:
    all_result_tokens = []
    for word_str, _offsets in pre_tokenized:
        tokens = list(word_str)
        for a, b in merges[:num_merges]:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        all_result_tokens.extend(tokens)

    tok_display = " | ".join(
        f"{C.GREEN}{decode_token(t)!r}{C.RESET}" for t in all_result_tokens
    )
    label = f"merges={num_merges:>5}"
    count = len(all_result_tokens)
    print(f"    {C.YELLOW}{label}{C.RESET}  ({count:2d} токенов):  [{tok_display}]")

print()
print(f"  {C.DIM}Видно как с каждой стадией merge-ов токены становятся длиннее —")
print(f"  токенизатор \"узнаёт\" всё более крупные куски слов.{C.RESET}")

print(f"\n{C.CYAN}{'═' * 70}")
print(f"  {C.BOLD}Готово!{C.RESET}")
print(f"{C.CYAN}{'═' * 70}{C.RESET}")
print()
print(f"  {C.DIM}Попробуйте другие модели:{C.RESET}")
print(f"    python bpe_pretrained.py {C.CYAN}openai-community/gpt2{C.RESET}")
print(f"    python bpe_pretrained.py {C.CYAN}meta-llama/Llama-2-7b-hf{C.RESET}")
print(f"    python bpe_pretrained.py {C.CYAN}mistralai/Mistral-7B-v0.1{C.RESET}")
print(f"    python bpe_pretrained.py {C.CYAN}bigscience/bloom-560m{C.RESET}")
print()
