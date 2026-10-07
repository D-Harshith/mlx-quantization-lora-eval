import csv
import json
import random
import sys
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd().parent
SRC = ROOT/"Dataset"/"issues_test.csv"
OUT = ROOT/"Dataset"/"eval.jsonl"
df = pd.read_csv(SRC)
df = df.fillna("")
sample = df.groupby("label").sample(n=100, random_state=42)
print(len(sample))
print(sample["label"].value_counts())
print(sample["repo"].value_counts())
with open(OUT, "w", encoding="utf-8") as f:
    for i, row in sample.iterrows():
        case = {
            "id": int(i),   # original row number; int() because JSON can't store numpy numbers
            "repo": row["repo"],
            "expected": row["label"],
            "title": row["title"].strip(),
            "body": row["body"].replace("\r\n", "\n").strip()[:1500],
        }
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


