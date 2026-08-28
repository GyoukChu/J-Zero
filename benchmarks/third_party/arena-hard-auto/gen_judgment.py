import json
import yaml
import argparse
import os
import concurrent.futures

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS


# Verdict labels show_result.py's label_to_score can map. A regex can match a
# syntactically-valid but semantically-unknown label (e.g. "A>>>B", "<>"); mapping
# that as a score KeyErrors in show_result. We null those so the null-filter drops
# the game cleanly (carried-over from sibling's CW verdict-validity fix).
VALID_VERDICT_LABELS = frozenset(
    {"A>B", "A>>B", "A=B", "A<<B", "A<B", "B>A", "B>>A", "B=A", "B<<A", "B<A"}
)


def get_score(judgment, patterns):
    import re
    label = None
    for pattern in patterns:
        pattern = re.compile(pattern)

        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]

        if len(set(matches)) > 0:
            label = matches[-1].strip("\n")
            break
    if label is None or label not in VALID_VERDICT_LABELS:
        return None
    return label


def pairwise_judgment(question, baseline, answer, reference, configs, settings):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        # the judge that actually scored this category (per-category routing)
        "judge": args.get('judge_name', args['configs']['judge_model']),
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, reference: {configs["reference"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    if configs["reference"]:
        assert not configs["reference"] in models, "ERROR: one of the models being evaluated is used as reference."
        ref_answers = [answer_dir[model] for model in configs["reference"]]
    else:
        ref_answers = None
    
    # ------------------------------------------------------------------
    # Per-category judge routing. Each question's category selects its judge
    # (JUDGE_SETTINGS[category]["judge"], e.g. hard_prompt -> qwen36-judge,
    # creative_writing -> gemma4-judge), falling back to the setting file's
    # global judge_model. Judgments are written under the routed judge's dir
    # (model_judgment/<judge>/<model>.jsonl), so a single model's judgments are
    # split across judge dirs by category. show_result.py loads each category
    # from its own judge dir.
    # ------------------------------------------------------------------
    default_judge = configs["judge_model"]

    def judge_for(category):
        return JUDGE_SETTINGS.get(category, {}).get("judge", default_judge)

    # judges actually needed for the categories present in the question set
    needed_judges = sorted({judge_for(q["category"]) for q in questions})
    for j in needed_judges:
        assert j in endpoint_list, (
            f"judge '{j}' has no entry in the endpoint file — add it to "
            f"config/api_config.yaml (or set JUDGE_SETTINGS[cat]['judge'])."
        )
    print(f"per-category judges: " + ", ".join(
        f"{c}->{judge_for(c)}" for c in sorted({q['category'] for q in questions})))

    # output file per (judge, model) + existing judgments per judge dir
    output_files = {}          # (judge, model) -> path
    existing_judgments = {}     # judge -> {model: {uid: record}}
    for judge in needed_judges:
        jdir = f"data/{configs['bench_name']}/model_judgment/{judge}"
        os.makedirs(jdir, exist_ok=True)
        existing_judgments[judge] = load_model_answers(jdir)
        for model in models:
            output_files[(judge, model)] = os.path.join(jdir, f"{model}.jsonl")

    max_workers = max(endpoint_list[j]["parallel"] for j in needed_judges)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                uid = question["uid"]
                category = question["category"]
                judge = judge_for(category)

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not uid in model_answers[model]:
                    print(f"Warning: {model} answer to {question['uid']} cannot be found.")
                    continue

                if model in existing_judgments[judge] and uid in existing_judgments[judge][model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][uid]
                kwargs["baseline"] = model_answers[
                    JUDGE_SETTINGS[category]["baseline"]
                ][uid]

                if ref_answers:
                    kwargs["reference"] = [ref_answer[uid] for ref_answer in ref_answers]
                else:
                    kwargs["reference"] = None

                kwargs["configs"] = configs
                kwargs["settings"] = endpoint_list[judge]
                kwargs["judge_name"] = judge
                kwargs["output_file"] = output_files[(judge, model)]

                future = executor.submit(judgment, kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()