"""
BPE-токенизатор из готовых merge-правил (merges.txt)
=====================================================
Загружает merge-правила из файла и токенизирует входной текст.

Использование:
  python tokenize_text.py "Hello, world!"
  python tokenize_text.py --file input.txt
  echo "some text" | python tokenize_text.py --stdin
"""

import sys
import re
import argparse

# ─── Цвета ───────────────────────────────────────────────
class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    RESET   = "\033[0m"

# ─── GPT-2 ByteLevel маппинг ────────────────────────────
def _bytes_to_unicode():
    """Стандартный bytes_to_unicode маппинг из GPT-2."""
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
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# ─── Pre-tokenization (GPT-2 pattern) ───────────────────
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
)

def pre_tokenize(text):
    """Разбивает текст на слова по паттерну GPT-2."""
    return GPT2_PAT.findall(text)

def encode_word_to_bpe_chars(word):
    """Кодирует строку в ByteLevel символы (как GPT-2)."""
    return "".join(BYTE_ENCODER[b] for b in word.encode("utf-8"))

def decode_bpe_token(token):
    """Декодирует BPE-токен обратно в строку."""
    raw_bytes = bytes(BYTE_DECODER[c] for c in token)
    return raw_bytes.decode("utf-8", errors="replace")

# ─── Загрузка merge-правил ──────────────────────────────
def load_merges(path):
    merges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges

def build_merge_ranks(merges):
    """Строит словарь (пара) -> приоритет. Меньше = применяется раньше."""
    return {pair: i for i, pair in enumerate(merges)}

# ─── BPE-токенизация одного слова ────────────────────────
def get_pairs(tokens):
    """Возвращает множество пар соседних токенов."""
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

def bpe_encode_word(word_bpe_chars, merge_ranks):
    """
    Применяет BPE merge-правила к слову (уже закодированному в ByteLevel символы).
    Возвращает список BPE-токенов.
    """
    tokens = list(word_bpe_chars)
    if len(tokens) <= 1:
        return tokens

    while True:
        pairs = get_pairs(tokens)
        if not pairs:
            break
        # breakpoint()
        # Находим пару с наименьшим рангом (самую приоритетную)
        best_pair = min(pairs, key=lambda p: merge_ranks.get(p, float("inf")))
        if best_pair not in merge_ranks:
            break  # Ни одна пара не найдена в merge-правилах

        a, b = best_pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        if len(tokens) == 1:
            break

    return tokens

# ─── Полная токенизация ──────────────────────────────────
def tokenize(text, merge_ranks):
    """Токенизирует текст: pre-tokenize → ByteLevel → BPE merge."""

    words = pre_tokenize(text)
    print(f'PRETOKENIZED: {words}')
    all_tokens = []
    for word in words:
        bpe_chars = encode_word_to_bpe_chars(word)
        print(f'WORD: {word}, BPE_CHARS: {bpe_chars}')
        tokens = bpe_encode_word(bpe_chars, merge_ranks)
        print(f'TOKENS: {tokens}, BPE_CHARS: {bpe_chars}')

        all_tokens.extend(tokens)
    return all_tokens

def display_token(t):
    """Показывает токен в читаемом виде."""
    decoded = decode_bpe_token(t)
    return decoded.replace(" ", "·").replace("\n", "↵").replace("\t", "→")

# ─── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BPE-токенизатор из merges.txt")
    parser.add_argument("text", nargs="?", help="Текст для токенизации")
    parser.add_argument("-m", "--merges", default="merges.txt", help="Путь к merges.txt")
    parser.add_argument("-f", "--file", help="Читать текст из файла")
    parser.add_argument("--stdin", action="store_true", help="Читать из stdin")
    parser.add_argument("-v", "--verbose", action="store_true", help="Показать пошаговый процесс")
    args = parser.parse_args()

    # Определяем текст
    if args.stdin:
        text = sys.stdin.read()
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    # Загружаем merge-правила
    print(f"{C.DIM}Загрузка merge-правил из {args.merges}...{C.RESET}")
    merges = load_merges(args.merges)
    merge_ranks = build_merge_ranks(merges)
    print(f"{C.DIM}Загружено {len(merges)} merge-правил.{C.RESET}\n")

    # Токенизируем
    tokens = tokenize(text, merge_ranks)

    # Вывод результата
    print(f"{C.BOLD}Текст:{C.RESET}    {C.GREEN}{text!r}{C.RESET}")
    print(f"{C.BOLD}Токенов:{C.RESET}  {C.YELLOW}{len(tokens)}{C.RESET}\n")

    print(f"{C.BOLD}Токены:{C.RESET}")
    for i, t in enumerate(tokens):
        d = display_token(t)
        print(f"  {C.YELLOW}{i:4d}{C.RESET}  {C.GREEN}{d!r:<20}{C.RESET}  {C.DIM}(bpe: {t!r}){C.RESET}")

    # Проверяем что декодирование совпадает
    decoded = "".join(decode_bpe_token(t) for t in tokens)
    print(f"\n{C.BOLD}Декодировано:{C.RESET}  {C.CYAN}{decoded!r}{C.RESET}")

    if decoded == text:
        print(f"{C.GREEN}Roundtrip OK!{C.RESET}")
    else:
        print(f"{C.RED}Roundtrip MISMATCH!{C.RESET}")
        print(f"  Оригинал:      {text!r}")
        print(f"  Декодировано:   {decoded!r}")

    # Verbose: пошаговый процесс
    if args.verbose:
        print(f"\n{C.CYAN}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}Пошаговый процесс merge-ов:{C.RESET}")
        print(f"{C.CYAN}{'═' * 60}{C.RESET}\n")

        words = pre_tokenize(text)
        for word in words:
            bpe_chars = encode_word_to_bpe_chars(word)
            toks = list(bpe_chars)
            d = display_token(bpe_chars)
            print(f"  Слово: {C.MAGENTA}{d!r}{C.RESET}")
            print(f"    начало: [{', '.join(repr(display_token(c)) for c in toks)}]")

            step = 0
            while len(toks) > 1:
                pairs = get_pairs(toks)
                best = min(pairs, key=lambda p: merge_ranks.get(p, float("inf")))
                if best not in merge_ranks:
                    break

                a, b = best
                new_toks = []
                i = 0
                while i < len(toks):
                    if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                        new_toks.append(a + b)
                        i += 2
                    else:
                        new_toks.append(toks[i])
                        i += 1
                toks = new_toks
                step += 1
                rank = merge_ranks[(a, b)]
                tok_display = ", ".join(f"{C.GREEN}{display_token(t)!r}{C.RESET}" for t in toks)
                print(f"    шаг {step}: {display_token(a)!r} + {display_token(b)!r}"
                      f" → {C.BOLD}{display_token(a+b)!r}{C.RESET}"
                      f"  {C.DIM}(merge #{rank+1}){C.RESET}"
                      f"  => [{tok_display}]")
            print()


if __name__ == "__main__":
    main()
