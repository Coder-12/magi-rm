import json
from prompts import SOLVER_PROMPT, REVIEWER_PROMPT, REFINER_PROMPT


class Solver:
    def __init__(self, llm, k=4):
        self.llm = llm
        self.k = k

    def generate(self, question: str):
        prompt = SOLVER_PROMPT.format(question=question)
        outputs = self.llm.chat(prompt, n=self.k)
        return [{'id': i, 'text': o} for i, o in enumerate(outputs)]

class Reviewer:
    def __init__(self, llm, reward_model):
        self.llm = llm
        self.rm = reward_model

    def review(self, candidates):
        """
        For each candidate chain, produce:
        {candidate_id: id, bad_steps: [idx,...], feedback: <string>}
        The reviewer uses RM step scores to identify low-scoring steps and asks for targeted corrections.
        """
        reviews = []
        for c in candidates:
            # compute step scores from RM
            step_scores = self.rm.score_chain_steps(c['text'])
            # identify bad steps (score < threshold)
            bad_steps = [i for i, s in enumerate(step_scores) if s < 0.6] # heuristic
            # Build a review prompt that includes low-score step indices and a request for correction
            review_prompt = REVIEWER_PROMPT.format(candidates=c['text'], bad_steps=bad_steps, step_scores=step_scores)
            outputs = self.llm.chat(review_prompt, n=1)
            try:
                # Try JSON first (structured feedback)
                fb = json.loads(outputs[0])
                if isinstance(fb, dict) and 'feedback' in fb:
                    reviews.append({'candidate_id': c['id'], 'bad_steps': bad_steps, 'feedback': fb['feedback']})
                    continue
            except Exception:
                # fallback: use raw text as feedback
                pass
            reviews.append({'candidate_id': c['id'], 'bad_steps': bad_steps, 'feedback': outputs[0]})

        return reviews


class Refiner:
    def __init__(self, llm):
        self.llm = llm

    def refine(self, candidate, feedback):
        prompt = REFINER_PROMPT.format(candidate=candidate['text'], feedback=feedback)
        refined_text = self.llm.chat(prompt, n=1)[0]
        return {'id': candidate['id'], 'text': refined_text}