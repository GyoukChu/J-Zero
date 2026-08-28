import logging
import re
import statistics as stats
import random
import numpy as np
from typing import Dict, Any, List


SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20
def parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    scores = {}

    # Parse scores using multiple regex patterns
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    # Pattern 2: Metric: [Score]
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    # Combine both patterns
    matches1 = re.findall(score_pattern1, judge_model_response)
    matches2 = re.findall(score_pattern2, judge_model_response)
    
    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            score = float(match[1])
            # Add check to ensure score <= 20
            if score <= SCORE_RANGE_MAX:
                scores[metric_name] = score
            # If score > 20, it's discarded/ignored

    return scores

def invert_if_negative(metric: str, score: float, negative_criteria: List[str]) -> float:
    """
    If metric is a negative criterion, invert so that higher => better:
    e.g. 20 => 0, 0 => 20 =>  new_val = 20 - old_val
    """
    if metric in negative_criteria:
        return 20.0 - score
    #print(score)
    return score


def compute_creative_scores(tasks: List[Dict[str, Any]], negative_criteria: List[str]) -> float:
    """
    Each “task” can have multiple seed modifiers, each with judge_scores.  
    We'll gather all numeric scores, invert negative metrics, average them, 
    then that is each piece’s final. Then average across all pieces => 0..20 scale.
    """
    piece_scores = []
    for task_data in tasks:
        # results_by_modifier => { seedMod: {"model_response":..., "judge_scores":{}} }
        rdict = task_data.get("results_by_modifier", {})
        for seed_mod, block in rdict.items():
            j_scores = block.get("judge_scores", {})
            if not j_scores:
                continue
            # Gather valid numeric scores
            local_vals = []
            for metric, val in j_scores.items():
                if isinstance(val, (int, float)):
                    new_val = invert_if_negative(metric, val, negative_criteria)
                    if new_val <= SCORE_RANGE_MAX:
                        local_vals.append(new_val)
            if local_vals:
                piece_score = sum(local_vals) / len(local_vals)  # average 0..20
                piece_scores.append(piece_score)

    if not piece_scores:
        return 0.0
    #print('item score:', sum(piece_scores) / len(piece_scores) / 2)
    return sum(piece_scores) / len(piece_scores)


def compute_single_benchmark_score_creative(tasks, negative_criteria):
    """
    Returns a dict:
      {
        "overall_score": (0..20),
        "eqbench_creative_score": (0..100)
      }
    We produce eqbench_creative_score by scaling 0..20 => 0..10 => 0..100
    """
    avg_0_20 = compute_creative_scores(tasks, negative_criteria)
    # scale to 0..100    
    eqbench_score = avg_0_20 * 5.0
    eqbench_score = round(eqbench_score, 2)
    return {
        "overall_score": round(avg_0_20, 2),
        "eqbench_creative_score": eqbench_score
    }


def bootstrap_benchmark_stability_creative(tasks, negative_criteria, n_bootstrap=500, confidence_level=0.95):
    """
    Bootstraps the final overall_score from a sample of tasks. Return a dict with stats.
    """
    original_result = compute_single_benchmark_score_creative(tasks, negative_criteria)
    original_score = original_result["overall_score"]
    if not tasks:
        return {
            "error": "No tasks found for bootstrap"
        }

    # We'll treat each entire "task" as a sampling unit
    boot_scores = []
    for _ in range(n_bootstrap):
        sample_tasks = random.choices(tasks, k=len(tasks))
        sc = compute_single_benchmark_score_creative(sample_tasks, negative_criteria)
        boot_scores.append(sc["overall_score"])

    boot_scores.sort()
    lower_idx = int((1 - confidence_level)/2 * len(boot_scores))
    upper_idx = int((1 + confidence_level)/2 * len(boot_scores)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(upper_idx, len(boot_scores)-1)

    ci_lower = boot_scores[lower_idx]
    ci_upper = boot_scores[upper_idx]
    mean_ = np.mean(boot_scores)
    std_ = np.std(boot_scores, ddof=1)

    return {
        "original": original_score,
        "bootstrap_mean": float(mean_),
        "bootstrap_std": float(std_),
        "standard_error": float(std_),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap
    }
