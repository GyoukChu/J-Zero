import json
import re
import sys
import argparse
import glob
import os

# ============================================================================
# BBH eval — R-Zero-protocol generation with a two-layer scorer.
#
# Generation follows the em_rzero recipe (same prompt, chat template,
# temperature=0, max_tokens=8192, DP-by-question sharding, same output/.score
# schema); saved shards can be re-scored offline with --rescore.
#
# Layer 1 is stock BBEH fuzzy_match, kept verbatim, so everything stock scoring
# accepts stays accepted. It is near-exact string equality on the \boxed{}
# content, so on its own it fails correct-but-verbose answers
# (\boxed{(C) 2pm to 8pm} vs gold "(C)") — penalizing verbosity, not accuracy.
#
# Layer 2 runs only where layer 1 fails:
#  - Gold "(x)": accept a leading "(x) "/"x) "/"x. " marker, an "answer is x"
#    tail, or a UNIQUE "(x)" in the text; several distinct letters stay wrong.
#  - Gold single token (yes/no/valid/7/...): compare the prediction's FIRST
#    word with numeric equality, plus the yes/no alias groups defined below.
#  - Gold multi-token (word_sorting lists, dyck sequences): separator- and
#    whitespace-insensitive comparison.
#  - The full boxed content is used (layer 1 cuts at the first line).
#
# extract_last_boxed brace-counts (any nesting depth); .score sidecars also
# report "no_prediction". Scoring is deterministic — BBH never guessed randomly.
# ============================================================================

# BBH MCQ letters go up to (R) (reasoning_about_colored_objects).
_BBH_LETTER = 'a-r'


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


# --------------------------------------------------------------------------
# Layer 1: the ORIGINAL BBEH scoring, byte-identical behavior.
# --------------------------------------------------------------------------

def strip_latex(response: str) -> str:
  if response.startswith("$") and response.endswith("$"):
    response = response[1:-1]
  if "boxed{" in response and response.endswith("}"):
    response = response[0:-1].split("boxed{")[1]
  if "text{" in response and response.endswith("}"):
    response = response[0:-1].split("text{")[1]
  if "texttt{" in response and response.endswith("}"):
    response = response[0:-1].split("texttt{")[1]
  return response


def extract_answer(sample: str) -> str:
  if sample is None:
     sample = ""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is "
  ]
  answer = sample
  for answer_prefix in answer_prefixes:
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard (original)."""
  if prediction == reference:
    return True
  if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
    return prediction[1] == reference
  if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
    return reference[1] == prediction
  try:
    if float(prediction) == float(reference):
      return True
  except ValueError:
    pass
  if prediction.replace("'", "") == reference.replace("'", ""):
    return True
  if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
    return True
  if prediction.endswith("?") and prediction[:-1] == reference:
    return True
  return False


def preprocess_sample(sample: str) -> str:
    if sample is None:
        sample = ""
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


# --------------------------------------------------------------------------
# Layer 2: lenient matching, applied only when the original scoring fails.
# --------------------------------------------------------------------------

_WRAPPER_RE = re.compile(r'\\(?:text|textbf|texttt|textrm|textit|mathrm|mathbf|boxed)\{(.*)\}', re.S)
_PREFIX_RE = re.compile(
    r'^(?:the\s+)?(?:correct\s+|final\s+|best\s+|right\s+)?'
    r'(?:answer|option|choice|letter)\s*(?:is\s*)?:?\s*', re.I)

# Alias groups for yes/no-style golds, applied ONLY when both the gold and the
# prediction's first word are inside a group (never for e.g. "valid"/"invalid").
_YES = {'yes', 'true', 'plausible'}
_NO = {'no', 'false', 'implausible', 'unplausible'}


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


def _normalize_pred(sample):
    """Lowercased, wrapper/markdown-stripped, whitespace-collapsed prediction
    over the FULL extracted string (no first-line truncation)."""
    s = _strip_wrappers(str(sample)).replace('**', '')
    s = re.sub(r'\s+', ' ', s).strip().lower()
    s = s.strip('"\'').strip()
    if s.endswith('.'):
        s = s[:-1].strip()
    return s


def _sep_normalize(s):
    return re.sub(r'\s+', ' ', re.sub(r'[,\n;]+', ' ', s)).strip()


def lenient_match(sample, reference) -> bool:
    if sample is None:
        return False
    ref = preprocess_reference(str(reference))
    s = _normalize_pred(sample)
    if not s:
        return False
    stripped = _PREFIX_RE.sub('', s)

    # ---- gold is an MCQ letter "(x)" ----
    m_ref = re.fullmatch(r'\(([%s])\)' % _BBH_LETTER, ref)
    if m_ref:
        letter = m_ref.group(1)
        for cand in (s, stripped):
            m = re.match(r'^\(([%s])\)' % _BBH_LETTER, cand) or \
                re.match(r'^([%s])\)(?:\s|$)' % _BBH_LETTER, cand) or \
                re.match(r'^([%s])[.:](?:\s|$)' % _BBH_LETTER, cand)
            if m:
                return m.group(1) == letter
        m = re.search(r'(?:answer|option|choice)\s*(?:is)?\s*:?\s*'
                      r'\(?([%s])\)?\s*[.!]?\s*$' % _BBH_LETTER, s)
        if m:
            return m.group(1) == letter
        found = set(re.findall(r'\(([%s])\)' % _BBH_LETTER, s))
        if len(found) == 1:
            return found.pop() == letter
        return False

    # ---- gold is a single token (yes / no / valid / True / 7 / ...) ----
    if ' ' not in ref:
        tok = re.split(r'[\s,.:;!?]+', stripped or s, 1)[0]
        if tok == ref:
            return True
        try:
            if float(tok) == float(ref):
                return True
        except ValueError:
            pass
        if ref in _YES and tok in _YES:
            return True
        if ref in _NO and tok in _NO:
            return True
        return False

    # ---- gold is multi-token (word lists, bracket sequences) ----
    if _sep_normalize(s) == _sep_normalize(ref):
        return True
    if re.sub(r'\s', '', s) == re.sub(r'\s', '', ref):
        return True
    return False


def evaluate_correctness(sample: str, reference: str) -> bool:
    # Layer 1: original BBEH scoring — anything it accepted stays accepted.
    prediction = preprocess_sample(sample)
    ref = preprocess_reference(reference)
    if fuzzy_match(prediction, ref):
        return True
    # Layer 2: lenient scoring of verbose-but-unambiguous answers.
    return lenient_match(sample, reference)


# --------------------------------------------------------------------------

def load_bbh(bbh_dir):
    """Build the {task,question,answer} list from BBH json files."""
    rows = []
    for f in sorted(glob.glob(os.path.join(bbh_dir, "*.json"))):
        task = os.path.splitext(os.path.basename(f))[0]
        for ex in json.load(open(f))["examples"]:
            rows.append({"task": task, "question": ex["input"], "answer": ex["target"]})
    if not rows:
        raise FileNotFoundError(
            f"no BBH task files under {bbh_dir} — clone BIG-Bench-Hard per "
            "benchmarks/THIRD_PARTY.md (a missing dir would silently score 0)")
    return rows


def score_entries(entries):
    """Score saved entries -> (success, fail, no_prediction, per_category)."""
    success = fail = no_pred = 0
    per_category = {}
    for entry in entries:
        extracted = extract_solution(entry['solution'])
        cat = entry.get('task', '?')
        pc = per_category.setdefault(cat, [0, 0])
        if extracted is None:
            no_pred += 1
        if evaluate_correctness(extracted, entry['answer']):
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
    # benchmarks/ is this file's parent's parent (em_rzero/ -> benchmarks/).
    parser.add_argument("--bbh_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "third_party", "BIG-Bench-Hard", "bbh"))
    parser.add_argument("--results_file", type=str, default="final_results.jsonl")
    parser.add_argument("--tp", type=int, default=1)
    # DP-by-question sharding (mirrors the em_rzero evals). dp_total=1 => unsharded run.
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
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85)
    data = load_bbh(args.bbh_dir)
    categories = sorted(list(set(d['task'] for d in data)))
    print("Categories:", categories)

    print('----------------- Start Answering -------------------')

    all_entries = []
    for category in categories:
        for entry in data:
            if entry['task'] == category:
                all_entries.append((category, entry))
    shard_entries = [ce for i, ce in enumerate(all_entries) if i % args.dp_total == args.dp_shard]
    print(f"[shard {args.dp_shard}/{args.dp_total}] {len(shard_entries)}/{len(all_entries)} entries")

    prompts = []
    for category, entry in shard_entries:
        query = entry['question'] + '\n'
        messages = [{
            "role": "user",
            "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}.'
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
    ordered_pc = {c: per_category.get(c, [0, 0]) for c in categories}

    with open(args.output_file, 'w') as f:
        json.dump(answers, f, indent=2)
    with open(args.output_file + ".score", 'w') as f:
        json.dump({"success": success, "fail": fail, "no_prediction": no_pred,
                   "per_category": ordered_pc}, f)

    if args.dp_total == 1:
        with open(args.results_file, 'a') as f:
            json.dump({"dataset": "bbh", "model": args.model_path,
                       "accuracy": round(success / (success + fail) * 100, 2)}, f, indent=2)
        print("Overall Accuracy:", success / (success + fail))
    else:
        print(f"[shard {args.dp_shard}/{args.dp_total}] success={success} fail={fail} "
              f"no_prediction={no_pred}")