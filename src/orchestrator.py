import yaml
import numpy as np
import math
from collections import Counter, defaultdict
from scipy.special import softmax             # For weighted SC
from agents import Solver, Reviewer, Refiner
from llm_interface import LLMInterface
from reward_model import RewardModel
from statistics import mean


class MagicoreOrchestrator:
    def __init__(self, config_path='experiments/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.cfg = config
        self.llm = LLMInterface(model_name=config['llm']['model'], temperature=config['temperature'])
        self.solver = Solver(self.llm, k=config['solver_k'])
        self.rm = RewardModel(self.llm)
        self.reviewer = Reviewer(self.llm, self.rm)
        self.refiner = Refiner(self.llm)

    @staticmethod
    def extract_final_answer(text):
        """Extract the final answer string from a solver/refiner chain text."""
        if not text:
            return ""
        if 'Final Answer:' in text:
            return text.split('Final Answer:')[-1].strip()
        else:
            lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
            return lines[-1] if lines else ''

    @staticmethod
    def majority_vote(candidates):
        """Return (best_answer, vote_confidence, answers_list)."""
        answers = [MagicoreOrchestrator.extract_final_answer(c['text']) for c in candidates]
        cnt = Counter(answers)
        best_ans, count = cnt.most_common(1)[0]
        confidence = count / len(answers)
        return best_ans, confidence, answers

    def weighted_self_consistency(self, candidates, rm_scores):
        """
        Weighted Self-Consistency:
        - Group chains by final answer.
        - Use ORM scores (rm_scores) to weight each chain; aggregate weights per answer.
        - Return (best_answer, normalized_weight_of_best).
        """
        # Build mapping: answer -> list of scores for chains with that answer
        ans_to_scores = defaultdict(list)
        for c, score in zip(candidates, rm_scores):
            ans = self.extract_final_answer(c['text'])
            ans_to_scores[ans].append(score)

        # If there are no candidates, fallback
        if not ans_to_scores:
            return "", 0.0

        # Option A (paper-faithful): softmax over all chain scores, then sum per answer.
        # Flatten scores to compute softmax normalization across all chains
        all_scores = []
        all_scores_map = [] # tuple (answer, score)
        for ans, scores in ans_to_scores.items():
            for s in scores:
                all_scores.append(float(s))
                all_scores_map.append((ans, float(s)))

        if len(all_scores) == 0:
            return max(ans_to_scores.items(), key=lambda kv: len(kv[1]))[0], 1.0

        # softmax across all chain scores to get per-chain probability
        probs = softmax(np.array(all_scores), axis=0) # returns same-length array

        # sum probabilities
        ans_weights = defaultdict(float)
        for (ans, s), p in zip(all_scores_map, probs):
            ans_weights[ans] += float(p)

        # normalize and pick best
        total_weights = sum(ans_weights.values()) if ans_weights else 0.0
        best_ans = max(ans_weights, key=ans_weights.get)
        best_weight_norm = ans_weights[best_ans] / (total_weights + 1e-12)
        return best_ans, float(best_weight_norm)

    def compute_conditions(self, candidates):
        # Step 1: compute per-candidate RM solution-level score
        per_candidate_Srm = {}
        for c in candidates:
            step_scores = self.rm.score_chain_steps(c['text'])
            per_candidate_Srm[c['id']] = float(np.mean(step_scores)) if len(step_scores) > 0 else 0.0

        # Step 2: cluster by final answer
        answer_clusters = defaultdict(list)
        for c in candidates:
            ans = MagicoreOrchestrator.extract_final_answer(c['text'])
            answer_clusters[ans].append(c['id'])

        # Step 3: majority cluster
        maj_ans, maj_ids = max(answer_clusters.items(), key=lambda kv: len(kv[1]))

        S_rm_maj = np.mean([per_candidate_Srm[i] for i in maj_ids])
        S_rm_all = np.mean(list(per_candidate_Srm.values()))

        # --- Condition 1 (division-based normalization) ---
        if abs(S_rm_all) < 1e-8:
            S_rm_avg_norm = S_rm_maj - S_rm_all
        else:
            S_rm_avg_norm = (S_rm_maj / S_rm_all) - 1.0
        cond1 = (S_rm_avg_norm >= 0.0)

        # --- Condition 2: Reward model confidence via entropy + sigmoid ---
        cluster_scores = np.array([
            float(np.mean([per_candidate_Srm[i] for i in ids]))
            for _, ids in answer_clusters.items()
        ])

        # ensure positivity for stability
        if cluster_scores.min() < 0:
            # Shift if any negative due to LLM noise
            cluster_scores = cluster_scores - cluster_scores.min()
        cluster_scores = cluster_scores + 1e-8 # avoid zeros

        # linear normalization per paper (not softmax)
        p = cluster_scores / np.sum(cluster_scores)

        m = len(p)
        if m <= 1:
            conf_rm = 1.0
        else:
            H = -np.sum(p * np.log(p + 1e-12))
            H_max = math.log(m)
            H_norm = (H / (H_max + 1e-12))
            alpha = self.cfg.get('rm_conf_alpha', 5.0)
            conf_rm = 1.0 / (1.0 + np.exp(-alpha * (1.0 - H_norm))) # sigmoid(α*(1−H_norm))

        cond2 = (conf_rm >= self.cfg.get('rm_conf_threshold', 0.6))

        return cond1, cond2, S_rm_avg_norm, conf_rm

    def run(self, question: str):
        print("\n" + "=" * 80)
        print(f"[Question] {question}")
        print("=" * 80)

        # --- Coarse Stage ---
        print("\n[Stage 1] SOLVER generating initial reasoning chains...")
        coarse = self.solver.generate(question)
        print(f"  → Generated {len(coarse)} candidate chains.\n")
        coarse_ans, vote_conf, answers = MagicoreOrchestrator.majority_vote(coarse)
        # Ask RM for stepwise scores for each candidate (LLM-as-RM)
        cond1, cond2, S_rm_avg_norm, conf_rm = self.compute_conditions(coarse)

        print("[Orchestrator Check — Coarse Stage]")
        print(f"  Vote Confidence      : {vote_conf:.3f}")
        print(f"  Majority Quality Δ   : {S_rm_avg_norm:.3f}")
        print(f"  RM Confidence (σ)    : {conf_rm:.3f}")
        print(f"  Condition 1 (Maj OK) : {cond1}")
        print(f"  Condition 2 (Conf OK): {cond2}")
        print("-" * 80)

        # Accept coarse if both conditions + vote_conf satisfied
        if (vote_conf >= self.cfg['vote_threshold']) and (cond1 or cond2):
            print("[Decision] Coarse reasoning accepted ✅")
            print("=" * 80)
            return {
                'answer': coarse_ans,
                'method': 'coarse',
                'vote_conf': vote_conf,
                'S_rm_avg_norm': S_rm_avg_norm,
                'conf_rm': conf_rm,
            }
        print("[Decision] Refinement required 🔁\n")

        # --- Fine-grained Refinement ---
        candidates = coarse
        for i in range(self.cfg['max_iterations']):
            print("=" * 80)
            print(f"[Stage 2] Refinement Iteration {i + 1}")
            print("=" * 80)

            # Reviewer provides feedback
            print("\n[Reviewer] Evaluating candidate chains...")
            reviews = self.reviewer.review(candidates)
            print(f"  → {len(reviews)} reviews generated with step-wise feedback.\n")

            refined = []
            print("[Refiner] Applying localized feedback...")
            for r in reviews:
                cid = r.get('candidate_id')
                cand = next((c for c in candidates if c['id'] == cid), None)
                if cand:
                    new_cand = self.refiner.refine(cand, r['feedback'])
                    refined.append(new_cand)
                    print(f"    • Refined Candidate {cid}: feedback applied.")

            print(f"\n[Refiner] Produced {len(refined)} refined reasoning chains.\n")

            # Majority voting on refined outputs (Recompute metrics)
            ans, vote_conf, answers = self.majority_vote(refined)
            # Recompute easy/hard conditions after refinement
            cond1, cond2, S_rm_avg_norm, conf_rm = self.compute_conditions(refined)

            print(f"[Orchestrator Check — Iter {i + 1}]")
            print(f"  Vote Confidence      : {vote_conf:.3f}")
            print(f"  Majority Quality Δ   : {S_rm_avg_norm:.3f}")
            print(f"  RM Confidence (σ)    : {conf_rm:.3f}")
            print(f"  Condition 1 (Maj OK) : {cond1}")
            print(f"  Condition 2 (Conf OK): {cond2}")
            print(f"  Final Conf Threshold : {self.cfg['final_conf_threshold']}")
            print("-" * 80)

            # --- Final Confidence Check ---
            # After each refinement, re-score refined answers using RM to get overall RM confidence.
            # Compare with final_conf_threshold from config (same as in paper Section 4.2)
            if (cond1 or cond2) and (conf_rm >= self.cfg.get('final_conf_threshold', 0.0)):
                print(f"[Decision] Stopping — sufficient confidence reached ✅ (iter {i + 1})")
                print("=" * 80)
                return {
                    'answer': ans,
                    'method': f'fine_iter_{i+1}',
                    'vote_conf': vote_conf,
                    'S_rm_avg_norm': S_rm_avg_norm,
                    'conf_rm': conf_rm,
                }
            print("[Decision] Continuing to next refinement iteration...\n")
            # Retain top-k from coarse + refined via ORM (paper: ORM for retention)
            all_cands = coarse + refined
            orm_scores = [self.rm.aggregated_step_scores(self.rm.score_chain_steps(c['text']),
                                                         mode=self.cfg.get('rm_aggregation', 'mean')) for c in all_cands]
            top_indices = np.argsort(orm_scores)[-self.cfg['solver_k']:]  # Top-k by ORM
            candidates = [all_cands[i] for i in top_indices]

        print("[Decision] Max iterations reached — fallback to coarse ❌")
        print("=" * 80)
        return { # Use weighted SC on final candidates
            'answer': coarse_ans,
            'method': 'fallback',
            'vote_conf': vote_conf,
            'S_rm_avg_norm': S_rm_avg_norm,
            'conf_rm': conf_rm,
         }

if __name__ == '__main__':
    # Example
    import sys
    orch = MagicoreOrchestrator()
    prompt = sys.argv[1] if len(sys.argv)>1 else "If you have 12 apples and eat 3, how many are left?"
    res = orch.run(prompt)
    print(res)