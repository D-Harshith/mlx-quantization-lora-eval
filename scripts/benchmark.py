import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
COMPARISON = RESULTS / "comparison.csv"

parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=["4bit", "8bit", "bf16"])
parser.add_argument("--trials", type=int, default=10)
args = parser.parse_args()

# same as typing: mlx_lm.benchmark --model ... --prompt-tokens 400 --generation-tokens 200 --num-trials 10
command = [
    sys.executable, "-u", "-m", "mlx_lm", "benchmark",   # -u prints each trial as it finishes
    "--model", f"mlx-community/Llama-3.2-3B-Instruct-{args.variant}",
    "--prompt-tokens", "400",
    "--generation-tokens", "200",
    "--num-trials", str(args.trials),
]

# run it, showing each line and keeping a copy to read the averages from
lines = []
process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
for line in process.stdout:
    print(line, end="")
    lines.append(line)
if process.wait() != 0:
    sys.exit("benchmark failed, nothing saved")

# the last line looks like: Averages: prompt_tps=1437.952, generation_tps=61.669, peak_memory=2.324
averages = [line for line in lines if line.startswith("Averages")][0]
numbers = dict(re.findall(r"(\w+)=([\d.]+)", averages))

result = {
    "variant": args.variant,
    "generation_tps": round(float(numbers["generation_tps"]), 1),
    "prompt_tps": round(float(numbers["prompt_tps"]), 1),
    "trials": args.trials,
}
(RESULTS / f"{args.variant}_benchmark.json").write_text(json.dumps(result, indent=2))

# put the two numbers in this variant's row of the comparison table
df = pd.read_csv(COMPARISON)
row = df["variant"] == args.variant
df.loc[row, "generation_tps"] = result["generation_tps"]
df.loc[row, "prompt_tps"] = result["prompt_tps"]
df.to_csv(COMPARISON, index=False)
print(df[["variant", "accuracy", "generation_tps", "prompt_tps"]])
