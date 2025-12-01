import os
import re
import json
import time
from pathlib import Path

INPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_processed.jsonl"

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def append_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    # nQ = 0
    # for q in load_jsonl(INPUT_FILE):
    #     for i in range(len(q['generated'])):
    #         del q['generated'][i]['step_labels']
    #         if q['generated'][i]['timestamps']['labeled_at'][-1] == 'Z':
    #             q['generated'][i]['timestamps']['labeled_at'] = q['generated'][i]['timestamps']['labeled_at'][:-1]
    #
    #     if q['created_at'][-1] == 'Z':
    #         q['created_at'] = q['created_at'][:-1]
    #     if q['labeled_at'][-1] == 'Z':
    #         q['labeled_at'] = q['labeled_at'][:-1]
    #     append_jsonl(OUTPUT_FILE, q)
    #     nQ += 1
    # print(f"{nQ} questions processed and appended to {OUTPUT_FILE}")


    for q in load_jsonl(INPUT_FILE):
        print(q)
        break
    print("=========================")
    for q in load_jsonl(OUTPUT_FILE):
        print(q)
        break


if __name__ == "__main__":
    main()