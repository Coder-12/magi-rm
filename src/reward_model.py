# Two RM modes: 1) LLM-as-RM (prompt-based scoring). 2) Trained stepwise RM (PyTorch classifier/regressor)

import json, numpy as np
from prompts import RM_PROMPT

import logging
logger = logging.getLogger(__name__)

class RewardModel:
    def __init__(self, llm_interface):
        self.llm = llm_interface

    def _parse_steps_from_chain(self, chain_text):
        # Use same STEP_RE as in collection pipeline (numbered steps)
        import re
        STEP_RE = re.compile(r'^\s*\d+\.\s*(.*)', flags=re.MULTILINE)
        steps = STEP_RE.findall(chain_text)
        if not steps:
            # fallback: non-empty lines as coarse steps
            lines = [ln.strip() for ln in chain_text.splitlines() if ln.strip()]
            # drop possible heading lines like "Solution:"
            steps = lines
        return steps

    def score_chain_steps(self, chain_text: str):
        """
        Prompt the LLM to assign 0–10 correctness scores for each reasoning step (normalized to 0-1).
        Returns a list of floats, one per step.
        """
        prompt = RM_PROMPT.format(chain=chain_text)
        for attempt in range(3):
            response = self.llm.chat(prompt, n=1)[0]
            try:
                scores = json.loads(response)
                if isinstance(scores, list) and all(isinstance(s, (int, float)) for s in scores):
                    # Normalize 0-10 to 0-1
                    normalized = [max(0.0, min(1.0, float(s) / 10.0)) for s in scores]
                    # If lengths mismatch, try to adapt/truncate/pad
                    steps = self._parse_steps_from_chain(chain_text)
                    if len(normalized) != len(steps):
                        # attempt to align by truncation/padding with mean
                        if len(normalized) > len(steps):
                            normalized = normalized[:len(steps)]
                        else:
                            pad_val = float(np.mean(normalized)) if normalized else 0.5
                            normalized += [pad_val] * (len(steps) - len(normalized))
                    return normalized
            except Exception as e:
                logger.debug("RM parse attempt failed: %s", str(e))
                if attempt == 2:
                    logger.warning("RewardModel JSON parsing failed for RM response; falling back to heuristic.")
                    break # Final fail after retries

        # fallback path: produce uniform 0.5 for each parsed step (JSON fails)
        steps = self._parse_steps_from_chain(chain_text)
        return [0.5 for _ in steps]

    def aggregated_step_scores(self, step_scores: list, mode: str = 'mean'):
        """Aggregate step scores to solution-level (for ORM proxy)."""
        if mode == 'product':
            return float(np.prod(step_scores)) if step_scores else 0.0
        else: # default 'mean'
            return float(np.mean(step_scores)) if step_scores else 0.0
