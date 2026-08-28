#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score IFEval with the official instruction_following_eval scorer.

Computes the four standard metrics: prompt-level and instruction-level
accuracy, each strict and loose. The official scorer matches responses to
inputs by exact prompt string; we build that mapping from the prepared file
(id -> prompt) joined with the responses (id -> raw).
"""
import argparse, json, os, sys
from pathlib import Path

DEFAULT_PKG = str(Path(__file__).resolve().parents[2] / "third_party")
DEFAULT_INPUT = (DEFAULT_PKG +
                 "/instruction_following_eval/data/input_data.jsonl")


def acc(outputs):
    prompt_correct = sum(o.follow_all_instructions for o in outputs)
    instr_total = sum(len(o.follow_instruction_list) for o in outputs)
    instr_correct = sum(sum(o.follow_instruction_list) for o in outputs)
    return (100.0 * prompt_correct / len(outputs) if outputs else 0.0,
            100.0 * instr_correct / instr_total if instr_total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input_data", default=DEFAULT_INPUT)
    ap.add_argument("--pkg_dir", default=DEFAULT_PKG)
    args = ap.parse_args()

    sys.path.insert(0, args.pkg_dir)
    from instruction_following_eval import evaluation_lib

    id_to_prompt = {}
    with open(args.prepared) as fh:
        for line in fh:
            r = json.loads(line)
            id_to_prompt[r["id"]] = r["prompt"]
    prompt_to_response = {}
    with open(args.responses) as fh:
        for line in fh:
            r = json.loads(line)
            p = id_to_prompt.get(r["id"])
            if p is not None:
                prompt_to_response[p] = r["raw"]

    inputs = evaluation_lib.read_prompt_list(args.input_data)
    # The official scorer indexes prompt_to_response[inp.prompt] and KeyErrors
    # on a miss; treat a missing response as empty (i.e. follows nothing).
    n_missing = 0
    for inp in inputs:
        if inp.prompt not in prompt_to_response:
            prompt_to_response[inp.prompt] = ""
            n_missing += 1
    strict = [evaluation_lib.test_instruction_following_strict(i, prompt_to_response)
              for i in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(i, prompt_to_response)
             for i in inputs]

    ps, is_ = acc(strict)
    pl, il = acc(loose)
    summary = {
        "benchmark": "IFEval",
        "metric": "instruction_following (official)",
        "n": len(inputs),
        "n_missing_response": n_missing,
        "prompt_level_strict_acc": ps,
        "instruction_level_strict_acc": is_,
        "prompt_level_loose_acc": pl,
        "instruction_level_loose_acc": il,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary}, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()