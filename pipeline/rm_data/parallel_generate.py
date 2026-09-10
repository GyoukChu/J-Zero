"""Run independent vLLM generation shards on the visible GPUs."""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor


def _generate(gpu, model_path, chats, seed, mem, sampling, model_kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Give each shard its own port range: vLLM probes VLLM_PORT upward, so
    # concurrent shards never race for the same OS-assigned free port.
    os.environ["VLLM_PORT"] = str(int(os.environ.get("VLLM_PORT", "31000")) + 1000 * shard)
    os.environ.pop("MASTER_PORT", None)
    import vllm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = vllm.LLM(model=model_path, tokenizer=model_path,
                   tensor_parallel_size=1, gpu_memory_utilization=mem,
                   seed=seed, **model_kwargs)
    if tok.chat_template:
        prompts = [tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            add_special_tokens=True) for chat in chats]
    else:
        prompts = ["\n".join(f"{m['role']}: {m['content']}" for m in chat)
                   for chat in chats]
    sp = vllm.SamplingParams(stop_token_ids=[tok.eos_token_id], **sampling)
    responses = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    if len(responses) != len(chats):
        raise RuntimeError("vLLM returned an incomplete generation shard")
    return [[o.text for o in r.outputs] for r in responses]


def generate(model_path, chats, *, seed, mem, n, max_tokens, temperature,
             top_p, **model_kwargs):
    if not chats:
        return []
    gpus = [gpu.strip() for gpu in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")]
    if any(not gpu for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("CUDA_VISIBLE_DEVICES must list distinct GPUs")
    gpus = gpus[:len(chats)]
    sampling = dict(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    # A fresh process per shard releases its engine before the next model pass.
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        futures = [pool.submit(
            _generate, gpu, model_path,
            chats[len(chats) * i // len(gpus):len(chats) * (i + 1) // len(gpus)],
            int(seed) + i, mem, sampling, model_kwargs)
            for i, gpu in enumerate(gpus)]
        return [row for future in futures for row in future.result()]
