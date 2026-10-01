# utilities for evaluation (GSM8K, MATH, etc.)

from datasets import load_dataset
import re

NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

def normalize_answer(ans):
    # simple numeric extract
    m = NUMBER_RE.search(ans)
    if m:
        return float(m.group())
    return ans.strip().lower()

def evaluate_predictions(preds, golds):
    correct = 0
    for p,g in zip(preds,golds):
        if isinstance(g, float):
            try:
                pnum = normalize_answer(p)
                if abs(pnum-g) < 1e-6:
                    correct += 1
            except:
                pass
        else:
            if str(p).strip().lower() == str(g).strip().lower():
                correct += 1
    return correct / len(preds)

if __name__ == '__main__':
    ds = load_dataset('gsm8k', 'main')
    print(len(ds['train']))