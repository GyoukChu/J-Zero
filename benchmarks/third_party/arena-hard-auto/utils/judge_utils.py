OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

# Per-category judge routing for Auto-Arena-Hard-v2.0.
#   hard_prompt / coding / math  -> qwen36-judge   (Qwen3.6-27B, no-thinking)
#   creative_writing             -> gemma4-judge   (google/gemma-4-31b-it)
# `judge` names an entry in config/api_config.yaml (endpoints); it selects WHICH
# judge model scores that category. `baseline` is the opponent answer set the
# model is compared against; `system_prompt` is the judge instruction. If a
# category omits `judge`, gen_judgment falls back to the setting file's global
# `judge_model`.
JUDGE_SETTINGS = {
    "hard_prompt": {
        "judge": "qwen36-judge",
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "judge": "qwen36-judge",
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "judge": "qwen36-judge",
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "judge": "gemma4-judge",
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    },
    "arena-hard-v0.1": {
        "judge": "qwen36-judge",
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}