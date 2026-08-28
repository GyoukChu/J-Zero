import json
import re
import sys
import argparse

# ============================================================================
# MMLU-Pro eval — the official R-Zero script with fixed answer extraction.
# Generation is identical to the official script (same prompt, chat template,
# temperature=0, max_tokens=8192, DP-by-question sharding, same output/.score
# schema), so scores stay comparable and saved shards can be re-scored offline.
#
# Answer-extraction spec (differs from the official script):
#  - A letter counts only where it is an unambiguous choice marker (see
#    _letter_from_text). The original scanned 'A'..'J' as substrings, so any
#    capital inside a verbose box won, biased toward early letters.
#  - A box holding the option CONTENT (\boxed{10} for "(C): 10") is matched
#    against the options list; the match must be unique.
#  - No random fallback: unparseable predictions are deterministically wrong
#    and reported as "no_prediction" in the .score sidecar.
#  - extract_last_boxed brace-counts (any nesting depth);
#    extract_last_final_answer also matches at end-of-text.
#  - --rescore shard1.json ... re-scores saved outputs offline (no GPU) into
#    <shard>.rescore.score sidecars in the same schema.
# ============================================================================

LETTERS = 'ABCDEFGHIJ'


def extract_last_boxed(text):
    """Content of the last balanced \\boxed{...} (brace-counted, any depth)."""
    out = None
    i = 0
    marker = '\\boxed{'
    while True:
        j = text.find(marker, i)
        if j == -1:
            break
        k = j + len(marker)
        depth = 1
        start = k
        while k < len(text) and depth:
            if text[k] == '{':
                depth += 1
            elif text[k] == '}':
                depth -= 1
            k += 1
        if depth == 0:
            content = text[start:k - 1].strip()
            if content:
                out = content
        i = j + len(marker)
    return out


_FINAL_ANSWER_RES = [
    re.compile(r'final answer\s*(?:is)?\s*:?\s*(.+?)\s*(?:\n|$)', re.I),
    re.compile(r'the\s+(?:correct\s+|final\s+)?answer\s+is\s*:?\s*(.+?)\s*(?:\n|$)', re.I),
    re.compile(r'\banswer\s*:\s*(.+?)\s*(?:\n|$)', re.I),
]


def extract_last_final_answer(text):
    for rex in _FINAL_ANSWER_RES:
        matches = list(rex.finditer(text))
        if matches:
            return matches[-1].group(1)
    return None


def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant',
                              solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    for stop_word in ["</s>", "<|im_end|>", "<|endoftext|>"]:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    boxed = extract_last_boxed(model_output)
    if boxed:
        return boxed
    return extract_last_final_answer(model_output)


_WRAPPER_RE = re.compile(r'\\(?:text|textbf|texttt|textrm|textit|mathrm|mathbf|boxed)\{(.*)\}', re.S)
_PREFIX_RE = re.compile(
    r'^(?:the\s+)?(?:correct\s+|final\s+|best\s+|right\s+)?'
    r'(?:answer|option|choice|letter)\s*(?:is\s*)?:?\s*', re.I)


def _strip_wrappers(s):
    s = s.strip()
    while True:
        s2 = s.strip().strip('$').strip()
        m = _WRAPPER_RE.fullmatch(s2)
        if m:
            s2 = m.group(1).strip()
        if s2 == s:
            return s
        s = s2


def _letter_from_text(s):
    """A choice letter only where it is unambiguously a choice marker."""
    s = s.strip()
    # bare letter, possibly wrapped: C / (C) / [C] / C) / C. / C:
    m = re.fullmatch(r'[\(\[]?([A-Ja-j])[\)\]]?\s*[.:]?', s)
    if m:
        return m.group(1).upper()
    # leading marker then option text: "(C) 10", "C) 10", "C. 10", "C: 10"
    m = re.match(r'^[\(\[]([A-Ja-j])[\)\]]', s) or \
        re.match(r'^([A-Ja-j])\)\s', s) or \
        re.match(r'^([A-J])[.:]\s', s)
    if m:
        return m.group(1).upper()
    # trailing "... answer is C" / "... option (C)."
    m = re.search(r'(?:answer|option|choice)\s*(?:is)?\s*:?\s*'
                  r'[\(\[]?([A-Ja-j])[\)\]]?\s*[.!]?\s*$', s, re.I)
    if m:
        return m.group(1).upper()
    # a unique parenthesized letter anywhere in the text
    found = {c.upper() for c in re.findall(r'\(([A-Ja-j])\)', s)}
    if len(found) == 1:
        return found.pop()
    # "option C" / "choice B" anywhere, if unique
    found = {c.upper() for c in re.findall(r'\b(?:option|choice)\s*[\(\[]?([A-Ja-j])\b', s, re.I)}
    if len(found) == 1:
        return found.pop()
    return None


def _norm_text(s):
    s = re.sub(r'\s+', ' ', str(s).lower()).strip()
    s = s.strip('"\'').strip()
    if s.endswith('.'):
        s = s[:-1].strip()
    return s


def _squash(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _match_option_text(s, options):
    """If the prediction equals exactly one option's content, return its letter."""
    ns, qs = _norm_text(s), _squash(s)
    hits = []
    for i, opt in enumerate(options[:len(LETTERS)]):
        no, qo = _norm_text(opt), _squash(opt)
        hit = (ns and ns == no) or (qs and qs == qo)
        if not hit:
            try:
                hit = float(ns.replace(',', '')) == float(no.replace(',', ''))
            except (ValueError, AttributeError):
                pass
        if hit:
            hits.append(LETTERS[i])
    return hits[0] if len(hits) == 1 else None


def get_prediction(output, options=None):
    """Predicted letter, or None (counted wrong; never a random guess)."""
    solution = extract_solution(output)
    if solution is None:
        return None
    s = _strip_wrappers(solution).replace('**', '')
    s = re.sub(r'\s+', ' ', s).strip()
    letter = _letter_from_text(s)
    if letter is None:
        stripped = _PREFIX_RE.sub('', s)
        if stripped != s:
            letter = _letter_from_text(stripped)
    if letter is not None:
        return letter
    if options:
        return _match_option_text(s, options)
    return None


def score_entries(entries):
    """Score saved entries -> (success, fail, no_prediction, per_category)."""
    success = fail = no_pred = 0
    per_category = {}
    for entry in entries:
        prediction = get_prediction(entry['solution'], entry.get('options'))
        cat = entry.get('category', '?')
        pc = per_category.setdefault(cat, [0, 0])
        if prediction is None:
            no_pred += 1
        if prediction is not None and entry['answer'] == prediction:
            success += 1
            pc[0] += 1
        else:
            fail += 1
            pc[1] += 1
    return success, fail, no_pred, per_category


def report(success, fail, no_pred, per_category, tag=''):
    total = success + fail
    micro = success / total * 100 if total else 0.0
    valid = {c: v for c, v in per_category.items() if sum(v) > 0}
    macro = (sum(v[0] / (v[0] + v[1]) for v in valid.values()) / len(valid) * 100
             if valid else 0.0)
    print(f"{tag}n={total} success={success} no_prediction={no_pred}")
    print(f"{tag}Micro Average Accuracy: {micro:.2f}%")
    print(f"{tag}Macro Average Accuracy: {macro:.2f}%")
    return micro, macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model directory")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="File to save results")
    parser.add_argument("--tp", type=int, default=1)
    # DP-by-question sharding (identical to the official script). dp_total=1 => unsharded run.
    parser.add_argument("--dp_shard", type=int, default=0)
    parser.add_argument("--dp_total", type=int, default=1)
    parser.add_argument("--rescore", type=str, nargs='+', default=None,
                        help="Re-score existing output shard JSONs offline (no GPU).")
    args = parser.parse_args()

    # ---------------- offline rescore mode ----------------
    if args.rescore:
        tot_s = tot_f = tot_np = 0
        tot_pc = {}
        for path in args.rescore:
            with open(path) as f:
                entries = json.load(f)
            s, fl, np_, pc = score_entries(entries)
            with open(path + ".rescore.score", 'w') as f:
                json.dump({"success": s, "fail": fl, "no_prediction": np_,
                           "per_category": pc}, f)
            tot_s += s; tot_f += fl; tot_np += np_
            for c, (cs, cf) in pc.items():
                t = tot_pc.setdefault(c, [0, 0]); t[0] += cs; t[1] += cf
        report(tot_s, tot_f, tot_np, tot_pc, tag='[rescore] ')
        sys.exit(0)

    # ---------------- generation mode (unchanged pipeline) ----------------
    if not args.model_path:
        parser.error("--model_path is required unless --rescore is given")
    import datasets
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85)
    dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    def form_options(options: list):
        option_str = 'Options are:\n'
        for opt, o in zip(options, LETTERS):
            option_str += f'({o}): {opt}\n'
        return option_str

    print('----------------- Start Answering -------------------')

    all_entries = []
    for category in categories:
        for entry in dataset['test']:
            if entry['category'] == category:
                all_entries.append((category, entry))
    shard_entries = [ce for i, ce in enumerate(all_entries) if i % args.dp_total == args.dp_shard]
    print(f"[shard {args.dp_shard}/{args.dp_total}] {len(shard_entries)}/{len(all_entries)} entries")

    prompts = []
    for category, entry in shard_entries:
        query = entry['question'] + '\n' + form_options(entry['options']) + '\n'
        messages = [{
            "role": "user",
            "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
        }]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "user: " + query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params)

    answers = []
    for (category, entry), output in zip(shard_entries, outputs):
        entry['solution'] = output.outputs[0].text
        answers.append(entry)

    success, fail, no_pred, per_category = score_entries(answers)
    # keep the fixed category order (and zero rows) of the official script's sidecar
    ordered_pc = {c: per_category.get(c, [0, 0]) for c in categories}

    with open(args.output_file, 'w') as f:
        json.dump(answers, f, indent=2)
    with open(args.output_file + ".score", 'w') as f:
        json.dump({"success": success, "fail": fail, "no_prediction": no_pred,
                   "per_category": ordered_pc}, f)

    if args.dp_total != 1:
        print(f"[shard {args.dp_shard}/{args.dp_total}] success={success} fail={fail} "
              f"no_prediction={no_pred}")
        raise SystemExit(0)

    print("\n----- Accuracy Report -----")
    for category in categories:
        correct, incorrect = ordered_pc[category]
        total = correct + incorrect
        acc = correct / total if total else 0.0
        print(f"{category}: {correct}/{total} -> {acc*100:.2f}% accuracy")
    micro, macro = report(success, fail, no_pred, ordered_pc)
    with open('final_results.jsonl', 'a') as f:
        json.dump({"dataset": "mmlupro", "model": args.model_path,
                   "accuracy": round(micro, 2)}, f, indent=2)