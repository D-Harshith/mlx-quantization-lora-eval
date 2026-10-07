import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_lm import load, generate
from prompt import build_messages, parse_label

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "Dataset" / "eval.jsonl"
RESULTS = ROOT / "results"

parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=["4bit", "8bit", "bf16"])
parser.add_argument("--limit", type=int, help="run only N cases (smoke test)")
parser.add_argument("--adapter", help="folder with LoRA weights, e.g. ../adapters/4bit_lora")
parser.add_argument("--fused", help="folder with a fused model, e.g. ../models/4bit_lora_fused")

args = parser.parse_args()

cases = [json.loads(line) for line in open(EVAL, encoding="utf-8")]
random.Random(42).shuffle(cases)   # the file is sorted by label; mix it so --limit sees all three
if args.limit:
    cases = cases[:args.limit]

if args.fused:
    model, tokenizer = load(args.fused)   # the adapter is already merged into these weights
else:
    # adapter_path=None loads the plain base model
    model, tokenizer = load(f"mlx-community/Llama-3.2-3B-Instruct-{args.variant}", adapter_path=args.adapter)


def classify(case):
    prompt = tokenizer.apply_chat_template(build_messages(case), add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=5)   # greedy by default (temperature 0)


classify(cases[0])   # warm-up: the first call is slower, so don't time it

rows = []
for n, case in enumerate(cases, 1):
    start = time.perf_counter()
    raw = classify(case)
    seconds = time.perf_counter() - start
    predicted = parse_label(raw)
    rows.append({
        "id": case["id"],
        "repo": case["repo"],
        "expected": case["expected"],
        "predicted": predicted,
        "correct": predicted == case["expected"],
        "seconds": round(seconds, 3),
    })
    if n % 50 == 0:
        print(f"{n}/{len(cases)} done")
df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
variant = args.variant + ("_lora" if args.adapter else "")   # keep the base results, don't overwrite them
if args.fused:
    variant = args.variant + "_lora_fused"
name = variant + ("_smoke" if args.limit else "")
df.to_csv(RESULTS / f"{name}.csv", index=False)

summary = {
    "variant": variant,
    "cases": len(df),
    "accuracy": round(float(df["correct"].mean()), 3),
    "seconds_per_case": round(float(df["seconds"].mean()), 3),
    "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
}
for label, acc in df.groupby("expected")["correct"].mean().items():
    summary[f"accuracy_{label}"] = round(float(acc), 3)
(RESULTS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))