import json
from statistics import mean, stdev
from pathlib import Path

def load_jsonl(path):
    for line in open(path, "r", encoding="utf-8"):
        if line.strip():
            yield json.loads(line)

train = list(load_jsonl("data/splits/gsm8k_rm_train_FINAL_balanced.jsonl"))
low, mid, high = 0, 0, 0
for q in train:
    for c in q["generated"]:
        m = mean(c.get("step_scores_norm", [0]))
        if m < 0.3: low += 1
        elif m < 0.7: mid += 1
        else: high += 1

total = low + mid + high
print(f"Low:{100*low/total:.2f}%  Mid:{100*mid/total:.2f}%  High:{100*high/total:.2f}%  Total:{total}")
