# # Colab Batched Labeling + Checkpointing using Llama-3-8B-Instruct
# # Save as a single cell and run. Adjust INPUT_FILE/OUTPUT_FILE paths.
#
# # Install deps (run once)
# # !pip install -q transformers accelerate bitsandbytes datasets tqdm torch
#
# # Imports
# import os, json, time, math, random, re
# from pathlib import Path
# from tqdm.auto import tqdm
# from statistics import mean
# from datetime import datetime
# from collections import defaultdict
# from datasets import load_dataset
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#
# # -------- CONFIG --------
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# # Files on your mounted Drive (change these)
# INPUT_FILE = "./gsm8k_training_chains_1000_clean.jsonl"
# OUTPUT_FILE = "./gsm8k_training_chains_1000_llama_labeled.jsonl"
# CHECKPOINT_FILE = "./gsm8k_label_checkpoint.json"   # stores processed question ids / indices
# REPORT_FILE = "./gsm8k_label_report.json"
# # Batch / generation params
# BATCH_SIZE = 16               # chains per inference batch (tune for memory/runtime: T4 ~16-32)
# TEMPERATURE = 0.0             # deterministic
# MAX_NEW_TOKENS = 256
# DO_SAMPLE = False
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # Prompting / parsing
# STEP_JSON_PROMPT = """You are a scorer. Given a chain-of-thought broken into numbered steps, assign each step a score from 0 to 10 that estimates the correctness of that step.
# The maximum (best) score is 10, which means that this step is 100% correct (and 0% incorrect). The minimum (worst) score is 0, which means that this step is 100% incorrect (and 0% correct).
# Output a JSON dict in this exact format:
#
# {
#   "step_scores_raw": [s1, s2, ...],   // numbers 0..10
#   "chain_label": 0 or 1               // 1 if the chain final answer is correct vs. gold (if provided), else best-guess
# }
#
# Rules:
# - step_scores_raw are integers or floats 0..10 (10 = perfect).
# - Keep JSON minimal; nothing else in output (but we will robustly parse if extra text appears).
# - Be concise and avoid extra explanation.
#
# Question:
# {question}
#
# Chain:
# {chain}
# """
#
# # Safety & parsing helpers
# NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
# JSON_EXTRACT_RE = re.compile(r'\{.*\}', re.DOTALL)
#
# def normalize_numeric_string(s):
#     if s is None: return ""
#     s = str(s).replace(",", "").replace("$", "").strip()
#     m = NUMBER_RE.search(s)
#     if not m:
#         return s
#     try:
#         v = float(m.group(0))
#         return str(int(round(v))) if abs(v-round(v))<1e-9 else str(round(v,4))
#     except Exception:
#         return s
#
# def safe_parse_llm_json(text):
#     """Extract JSON object from model output; if fails, return None."""
#     m = JSON_EXTRACT_RE.search(text)
#     if not m:
#         return None
#     try:
#         return json.loads(m.group(0))
#     except Exception:
#         # attempt small fixes (replace single quotes etc.)
#         try:
#             txt = m.group(0).replace("'", '"')
#             return json.loads(txt)
#         except Exception:
#             return None
#
# # --------- Mount drive (Colab only) ----------
# try:
#     from google.colab import drive
#     drive.mount('/content/drive', force_remount=False)
# except Exception:
#     # Not in Colab or already mounted
#     pass
#
# # --------- Load model (transformers pipeline) ---------
# print("Loading tokenizer and model: this takes a minute or two...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token  # Set pad for batching
#
# # Load in 4-bit if possible (bitsandbytes)
# try:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         load_in_4bit=True,
#         trust_remote_code=True
#     )
#     print("Model loaded in 4-bit mode (bitsandbytes).")
# except Exception as e:
#     print("4-bit load failed, falling back to fp16 load. Error:", e)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True
#     )
#
# model.eval()
#
# # --------- Load gold answers for ORM ---------
# ds = load_dataset("gsm8k", "main")["train"]
# gold_dict = {ex["question"].strip(): normalize_numeric_string(ex["answer"].split("####")[-1].strip()) for ex in ds}
# print(f"Loaded {len(gold_dict)} gold answers for ORM labeling.")
#
# # --------- Read input questions (do not load entire file if extremely large) ---------
# input_path = Path(INPUT_FILE)
# if not input_path.exists():
#     raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
#
# # Read all records into list (for indexing and safe resume). This is okay for ~1000 records.
# records = []
# with open(input_path, "r", encoding="utf8") as fh:
#     for line in fh:
#         if line.strip():
#             records.append(json.loads(line))
#
# N = len(records)
# print(f"Loaded {N} records from {INPUT_FILE}")
#
# # --------- Checkpoint resume logic ---------
# checkpoint_path = Path(CHECKPOINT_FILE)
# if checkpoint_path.exists():
#     try:
#         ck = json.load(open(checkpoint_path, "r"))
#         processed_idxs = set(ck.get("processed_indices", []))
#         print(f"Resuming: found checkpoint with {len(processed_idxs)} processed indices.")
#     except Exception:
#         processed_idxs = set()
# else:
#     processed_idxs = set()
#
# # --------- Output files prepare (append mode used per batch) ----------
# out_path = Path(OUTPUT_FILE)
# out_path.parent.mkdir(parents=True, exist_ok=True)
# report_path = Path(REPORT_FILE)
#
# # --------- Batch processing loop ---------
# def process_batch(batch_indices):
#     """Process a batch of question indices and return list of labeled records + metadata."""
#     labeled_records = []
#     llm_calls = 0
#     batch_start_time = time.time()
#     all_prompts = []
#
#     # Collect prompts for all chains in batch
#     batch_chains = []  # list of (rec, chain) pairs
#     for idx in batch_indices:
#         rec = records[idx]
#         q = rec.get("question", "").strip()
#         for chain in rec.get("generated", []):
#             chain_text = chain.get("raw_text", "")
#             prompt = STEP_JSON_PROMPT.format(question=q, chain=chain_text)
#             all_prompts.append(prompt)
#             batch_chains.append((rec, chain))
#
#     # Run batched inference if prompts
#     if all_prompts:
#         inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 temperature=TEMPERATURE,
#                 do_sample=DO_SAMPLE,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         llm_calls = len(batch_outputs)
#
#     # Assign back
#     for j, (rec, chain) in enumerate(batch_chains):
#         q = rec.get("question", "").strip()
#         gold = gold_dict.get(q, None)
#         final_ans = normalize_numeric_string(chain.get("final_answer", ""))
#         # ORM: gold-based binary
#         chain_label = 1 if gold is not None and normalize_numeric_string(gold) == final_ans else 0
#
#         # PRM: parse step_scores from output
#         if j < len(batch_outputs):
#             out_text = batch_outputs[j]
#             parsed = safe_parse_llm_json(out_text)
#             if parsed and "step_scores_raw" in parsed:
#                 raw_scores = parsed["step_scores_raw"]
#                 step_scores_raw = [max(0.0, min(10.0, float(s))) for s in raw_scores]
#             else:
#                 step_scores_raw = [5.0] * len(chain.get("steps", []))  # neutral fallback
#         else:
#             step_scores_raw = [5.0] * len(chain.get("steps", []))  # neutral fallback
#
#         # Normalize and compute agg
#         step_scores_norm = [s/10.0 for s in step_scores_raw] if step_scores_raw else [0.0]
#         prm_mean = mean(step_scores_norm) if step_scores_norm else 0.0
#         prm_prod = math.prod([max(0.0, min(1.0, s)) for s in step_scores_norm]) if step_scores_norm else 0.0
#
#         # ORM score: use chain_label as binary, or refine if needed
#         orm_score = float(chain_label)
#
#         # Attach
#         chain["step_scores_raw"] = step_scores_raw
#         chain["step_scores_norm"] = step_scores_norm
#         chain["prm_score_mean"] = prm_mean
#         chain["prm_score_prod"] = prm_prod
#         chain["orm_score"] = orm_score
#         chain["chain_label"] = chain_label
#         chain["label_generated_at"] = datetime.utcnow().isoformat() + "Z"
#
#     batch_time = time.time() - batch_start_time
#     # Return unique labeled records (since multiple chains per rec)
#     unique_labeled = {id(rec): rec for rec, _ in batch_chains}.values()
#     return list(unique_labeled), llm_calls, batch_time
#
# # --------- Main loop with checkpointing ----------
# all_indices = list(range(N))
# to_process = [i for i in all_indices if i not in processed_idxs]
# print(f"Total to process: {len(to_process)} (BATCH_SIZE={BATCH_SIZE})")
#
# batch_count = math.ceil(len(to_process) / BATCH_SIZE)
# overall_start = time.time()
# total_llm_calls = 0
# processed = len(processed_idxs)
#
# for b in range(batch_count):
#     start = b * BATCH_SIZE
#     end = min(start + BATCH_SIZE, len(to_process))
#     batch_indices = to_process[start:end]
#     if not batch_indices:
#         continue
#
#     print(f"\n[Batch {b+1}/{batch_count}] processing indices {batch_indices[0]}..{batch_indices[-1]} ...")
#     labeled_batch, llm_calls, t_taken = process_batch(batch_indices)
#
#     # Append unique to output file
#     with open(out_path, "a", encoding="utf8") as outf:
#         for rec in labeled_batch:
#             outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
#
#     # Update checkpoint
#     for idx in batch_indices:
#         processed_idxs.add(idx)
#         processed += 1
#
#     checkpoint_data = {
#         "processed_indices": sorted(list(processed_idxs)),
#         "last_batch": {
#             "batch_num": b+1,
#             "batch_size": len(batch_indices),
#             "indices": [int(x) for x in batch_indices],
#             "timestamp": datetime.utcnow().isoformat() + "Z"
#         }
#     }
#     with open(checkpoint_path, "w", encoding="utf8") as cf:
#         json.dump(checkpoint_data, cf)
#
#     # Update cumulative stats & report
#     total_llm_calls += llm_calls
#     elapsed = time.time() - overall_start
#     eta_seconds = (elapsed / max(1, processed)) * (N - processed) if processed > 0 else 0
#     report = {
#         "processed_questions": processed,
#         "total_questions": N,
#         "processed_pct": round(100.0 * processed / N, 3),
#         "total_llm_calls": total_llm_calls,
#         "last_batch_time_s": t_taken,
#         "eta_minutes": round(eta_seconds / 60.0, 2),
#         "updated_at": datetime.utcnow().isoformat() + "Z",
#     }
#     with open(report_path, "w", encoding="utf8") as rf:
#         json.dump(report, rf, indent=2)
#
#     print(f"[Batch done] batch_size={len(batch_indices)} | llm_calls={llm_calls} | batch_time={t_taken:.1f}s")
#     print(f"Processed {processed}/{N} ({report['processed_pct']}%) | ETA {report['eta_minutes']} min")
#     # short sleep to be polite / avoid overheating
#     time.sleep(0.8)
#
# # Final report
# overall_time = time.time() - overall_start
# final_report = {
#     "finished": True,
#     "total_questions": N,
#     "processed_questions": processed,
#     "total_llm_calls": total_llm_calls,
#     "runtime_minutes": round(overall_time / 60.0, 2),
#     "completed_at": datetime.utcnow().isoformat() + "Z",
# }
# with open(report_path, "w", encoding="utf8") as rf:
#     json.dump(final_report, rf, indent=2)
#
# print("=== DONE ===")
# print(json.dumps(final_report, indent=2))