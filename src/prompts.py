# Prompt templates used by Solver, Reviewer, Refiner, and RM-scoring

# Prompt templates used by Solver, Reviewer, Refiner, and RM-scoring

SOLVER_PROMPT = """
You are a precise mathematical reasoning assistant.

Follow these rules carefully:

1. Solve the math problem **step by step**, numbering each step clearly (1., 2., 3., ...).
2. Substitute every number directly into equations before computing.
3. After each computation, explicitly check your arithmetic result.
4. After finishing, write one verification step that re-checks your final answer.
5. End your response with exactly one line:
   Final Answer: <numeric value>

Do not include any explanation after the final answer.

Problem:
{question}
"""

REVIEWER_PROMPT = """
You are a reviewer. You are given multiple candidate solutions with stepwise reward scores.
Your task:
1. Identify which steps are likely wrong.
2. Provide focused feedback to improve those steps.
3. Return a JSON list of dicts, each with keys: 'candidate_id', 'bad_steps', 'feedback'.

Candidates:
{candidates}

Stepwise scores:
{step_scores}
"""

REFINER_PROMPT = """
You are a refiner. Given the original candidate reasoning and targeted feedback, rewrite the reasoning with improvements.
Ensure corrected steps are concise, logical, and numbered consistently.
Finish with "Final Answer: <answer>".

Original candidate:
{candidate}

Feedback:
{feedback}
"""

RM_PROMPT = """
You are a scorer. Given a chain-of-thought broken into numbered steps, return a JSON array of numeric scores between 0 and 10 (higher = better) for each step *in order*. 
Example output -> [10, 8, 0, 9]

Important:
- Only output valid JSON (an array of numbers), nothing else.
- One number per step; the i-th number corresponds to the i-th enumerated step.
- If uncertain, return 5 for neutrality.

Chain:
{chain}
"""