import json, random
inp = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl"
out = "data/collected/gsm8k_training_chains_1000_labels_balanced.jsonl"

data = [json.loads(l) for l in open(inp)]
for q in data:
    pos = [c for c in q["generated"] if c["chain_label"] == 1]
    neg = [c for c in q["generated"] if c["chain_label"] == 0]
    keep_pos = random.sample(pos, min(len(pos), max(1, int(0.3 * len(pos+neg)))))
    q["generated"] = keep_pos + neg
with open(out, "w") as f:
    for q in data: f.write(json.dumps(q)+"\n")
