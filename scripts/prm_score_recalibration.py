import json, numpy as np
inp = "data/collected/gsm8k_training_chains_1000_labels_balanced.jsonl"
out = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm.jsonl"

def rescale(scores):
    arr = np.array(scores)
    mu, sigma = arr.mean(), arr.std() or 1
    norm = np.clip((arr - mu) / (2*sigma) + 0.5, 0, 1)
    return (norm * 10).tolist()

outf = open(out, "w")
for line in open(inp):
    q = json.loads(line)
    for c in q["generated"]:
        if "step_scores_raw" in c:
            c["step_scores_raw"] = rescale(c["step_scores_raw"])
            c["step_scores_norm"] = [s/10 for s in c["step_scores_raw"]]
    outf.write(json.dumps(q)+"\n")
outf.close()
