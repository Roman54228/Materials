import argparse
import json
import statistics
import time
from pathlib import Path

import requests

# Benchmark hygiene checklist:
# - Run with stable thermals and minimal background load.
# - Keep prompt/model/options fixed when comparing models.
# - Compare cold-start and warm-start separately.
# - Use enough runs to avoid one-off noise.


def safe_div(numerator, denominator):
    if denominator and denominator > 0:
        return numerator / denominator
    return None


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "p95": percentile(clean, 0.95),
        "stddev": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "max": max(clean),
    }


def format_stat(name, stats, unit=""):
    if not stats:
        return f"{name:<20} n/a"
    return (
        f"{name:<20} min={stats['min']:.4f}{unit} "
        f"mean={stats['mean']:.4f}{unit} median={stats['median']:.4f}{unit} "
        f"p95={stats['p95']:.4f}{unit} std={stats['stddev']:.4f}{unit}"
    )


def build_payload(args, prompt):
    return {
        "model": args.model,
        "prompt": prompt,
        "stream": args.stream,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "seed": args.seed,
        },
    }


def extract_metrics(base):
    prompt_tokens = base.get("prompt_eval_count", 0)
    prompt_time_s = base.get("prompt_eval_duration", 0) / 1e9
    gen_tokens = base.get("eval_count", 0)
    gen_time_s = base.get("eval_duration", 0) / 1e9
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_time_s": prompt_time_s,
        "gen_tokens": gen_tokens,
        "gen_time_s": gen_time_s,
        "load_time_s": base.get("load_duration", 0) / 1e9,
        "prompt_tok_s": safe_div(prompt_tokens, prompt_time_s),
        "gen_tok_s": safe_div(gen_tokens, gen_time_s),
    }


def run_once(args, payload):
    started = time.perf_counter()
    metrics = {}
    text_out = ""
    ttft_s = None

    if args.stream:
        with requests.post(args.url, json=payload, stream=True, timeout=args.timeout) as resp:
            resp.raise_for_status()
            final_chunk = None
            got_first_token = False
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if not got_first_token and chunk.get("response"):
                    ttft_s = time.perf_counter() - started
                    got_first_token = True
                text_out += chunk.get("response", "")
                if chunk.get("done"):
                    final_chunk = chunk
            if final_chunk is None:
                raise RuntimeError("Stream ended without final done chunk.")
            metrics = extract_metrics(final_chunk)
    else:
        resp = requests.post(args.url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        text_out = body.get("response", "")
        metrics = extract_metrics(body)
        ttft_s = metrics["load_time_s"]

    wall_time_s = time.perf_counter() - started
    return {
        "ok": True,
        "error": None,
        "response_text": text_out,
        "ttft_s": ttft_s,
        "wall_time_s": wall_time_s,
        **metrics,
    }


def validate_result(result, args):
    checks = []
    checks.append(("non_empty_response", bool(result.get("response_text", "").strip())))
    checks.append(("min_gen_tokens", result.get("gen_tokens", 0) >= args.min_gen_tokens))
    checks.append(("non_error_run", result.get("ok", False)))

    failed = [name for name, passed in checks if not passed]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def run_warmup(args, prompt):
    payload = build_payload(args, prompt)
    for idx in range(args.warmup):
        try:
            run_once(args, payload)
            print(f"[warmup {idx + 1}/{args.warmup}] ok")
        except Exception as exc:
            print(f"[warmup {idx + 1}/{args.warmup}] failed: {exc}")


def run_benchmark(args, prompt):
    payload = build_payload(args, prompt)
    results = []
    validations = []

    for i in range(args.runs):
        try:
            result = run_once(args, payload)
            validation = validate_result(result, args)
            result["validation_passed"] = validation["passed"]
            result["validation_failed_checks"] = validation["failed_checks"]
            print(
                f"[run {i + 1}/{args.runs}] ok "
                f"wall={result['wall_time_s']:.3f}s "
                f"gen_tok_s={result['gen_tok_s'] if result['gen_tok_s'] is not None else 'n/a'} "
                f"validation={'pass' if validation['passed'] else 'fail'}"
            )
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "response_text": "",
                "ttft_s": None,
                "wall_time_s": None,
                "load_time_s": None,
                "prompt_tok_s": None,
                "gen_tok_s": None,
                "prompt_tokens": 0,
                "gen_tokens": 0,
            }
            validation = {"passed": False, "failed_checks": ["request_failed"], "checks": []}
            print(f"[run {i + 1}/{args.runs}] failed: {exc}")

        results.append(result)
        validations.append(validation)

    return results, validations


def read_prompt(args):
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def print_summary(args, results, validations):
    successful = [r for r in results if r["ok"]]
    success_rate = safe_div(len(successful), len(results)) or 0.0
    validation_passed = sum(1 for v in validations if v["passed"])
    validation_rate = safe_div(validation_passed, len(validations)) or 0.0

    wall_stats = summarize([r.get("wall_time_s") for r in successful])
    ttft_stats = summarize([r.get("ttft_s") for r in successful])
    load_stats = summarize([r.get("load_time_s") for r in successful])
    prompt_stats = summarize([r.get("prompt_tok_s") for r in successful])
    gen_stats = summarize([r.get("gen_tok_s") for r in successful])

    print("\n=== Benchmark Summary ===")
    print(f"model:                {args.model}")
    print(f"url:                  {args.url}")
    print(f"stream:               {args.stream}")
    print(f"warmup runs:          {args.warmup}")
    print(f"measured runs:        {args.runs}")
    print(f"successful runs:      {len(successful)}/{len(results)} ({success_rate * 100:.1f}%)")
    print(f"validation pass rate: {validation_passed}/{len(validations)} ({validation_rate * 100:.1f}%)")
    print()
    print(format_stat("wall_time_s", wall_stats, "s"))
    print(format_stat("ttft_s", ttft_stats, "s"))
    print(format_stat("load_time_s", load_stats, "s"))
    print(format_stat("prompt_tok_s", prompt_stats, " tok/s"))
    print(format_stat("gen_tok_s", gen_stats, " tok/s"))

    if args.json_out:
        payload = {
            "config": {
                "url": args.url,
                "model": args.model,
                "stream": args.stream,
                "temperature": args.temperature,
                "num_predict": args.num_predict,
                "seed": args.seed,
                "warmup": args.warmup,
                "runs": args.runs,
                "timeout": args.timeout,
                "min_gen_tokens": args.min_gen_tokens,
            },
            "summary": {
                "success_rate": success_rate,
                "validation_rate": validation_rate,
                "wall_time_s": wall_stats,
                "ttft_s": ttft_stats,
                "load_time_s": load_stats,
                "prompt_tok_s": prompt_stats,
                "gen_tok_s": gen_stats,
            },
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json_out}")


def make_parser():
    parser = argparse.ArgumentParser(description="Reliable Ollama benchmark runner.")
    parser.add_argument("--url", default="http://localhost:11434/api/generate")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--prompt", default="Explain diffusion models shortly.")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--min-gen-tokens", type=int, default=1)
    parser.add_argument("--json-out", default=None)
    return parser


def main():
    args = make_parser().parse_args()
    prompt = read_prompt(args)

    if args.warmup > 0:
        print(f"Running warmup: {args.warmup} rounds (excluded from stats)")
        run_warmup(args, prompt)

    print(f"\nRunning benchmark: {args.runs} measured rounds")
    results, validations = run_benchmark(args, prompt)
    print_summary(args, results, validations)


if __name__ == "__main__":
    main()