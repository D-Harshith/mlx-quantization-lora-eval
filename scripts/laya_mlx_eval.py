import argparse
import json
import random
import time
from pathlib import Path

import laya_mlx as laya
import mlx.core as mx
import pandas as pd
from prompt import build_messages

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "Dataset" / "eval.jsonl"
RESULTS = ROOT / "results"

# same three labels and definitions as SYSTEM in prompt.py
QUESTION = {
    "label": {
        "type": "choice",
        "instructions": "Classify this GitHub issue.",
        "criteria": {
            "bug": "something is broken, crashes, or behaves differently from what is documented or expected",
            "feature": "a request for new functionality or an improvement to existing behavior",
            "question": "the author is asking for help, usage guidance, or clarification",
        },
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="aac6fef/laya-typed-decisions-mlx")
parser.add_argument("--limit", type=int, help="run only N cases (smoke test)")
args = parser.parse_args()

cases = [json.loads(line) for line in open(EVAL, encoding="utf-8")]
random.Random(42).shuffle(cases)   # same order as LLM_eval.py
if args.limit:
    cases = cases[:args.limit]

agent = laya.load(args.model)


def classify(case):
    issue_text = build_messages(case)[1]["content"]   # the same "Title: ... Body: ..." text the LLMs saw
    return agent.predict(issue_text, QUESTION)


classify(cases[0])   # warm-up: the first call is slower, so don't time it

rows = []
for n, case in enumerate(cases, 1):
    start = time.perf_counter()
    result = classify(case)
    seconds = time.perf_counter() - start
    answer = result["answers"]["label"]
    predicted = answer["choice"]
    rows.append({
        "id": case["id"],
        "repo": case["repo"],
        "expected": case["expected"],
        "predicted": predicted,
        "correct": predicted == case["expected"],
        "confidence": answer["probabilities"][predicted],
        "truncated": result["usage"]["truncated"],   # True if the issue didn't fit in Laya's 512-token window
        "seconds": round(seconds, 4),
    })
    if n % 50 == 0:
        print(f"{n}/{len(cases)} done")

df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
variant = args.model.split("/")[-1]   # e.g. laya-mlx
name = variant + ("_smoke" if args.limit else "")
df.to_csv(RESULTS / f"{name}.csv", index=False)

summary = {
    "variant": variant,
    "cases": len(df),
    "accuracy": round(float(df["correct"].mean()), 3),
    "seconds_per_case": round(float(df["seconds"].mean()), 4),
    "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    "truncated_cases": int(df["truncated"].sum()),
}
for label, acc in df.groupby("expected")["correct"].mean().items():
    summary[f"accuracy_{label}"] = round(float(acc), 3)
(RESULTS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))