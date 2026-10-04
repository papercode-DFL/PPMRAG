import argparse
import json
import os
import re
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def strip_tags(text):
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return re.sub(r"(?is)<[^>]+>", "", text).strip()


class LLM:
    def __init__(self, path, engine):
        from vllm import LLM as Engine
        self.engine = Engine(
            model=path, dtype="bfloat16", seed=0, trust_remote_code=True, enable_prefix_caching=True,
            max_model_len=engine["max_model_len"], max_num_seqs=engine["max_num_seqs"],
            gpu_memory_utilization=engine["gpu_memory_utilization"], allowed_local_media_path=engine["media_root"],
            limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": engine["max_pixels"]})
        self.repetition_penalty = engine["repetition_penalty"]

    def generate(self, system, prompts, images, max_tokens):
        """-> [(text, prompt + output tokens)]"""
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=self.repetition_penalty,
                                seed=0)
        convs = []
        for prompt, image in zip(prompts, images):
            user = prompt
            if image:
                user = [{"type": "image_url", "image_url": {"url": "file://" + os.path.abspath(image)}},
                        {"type": "text", "text": prompt}]
            convs.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
        outs = self.engine.chat(convs, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        return [(strip_tags(o.outputs[0].text), len(o.prompt_token_ids) + len(o.outputs[0].token_ids)) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="resolved run config (config.json in the run directory)")
    ap.add_argument("--model", required=True, help="key under models")
    ap.add_argument("--system", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.jobs, encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    llm = LLM(cfg["models"][args.model], cfg["vllm"])
    load_seconds = time.time() - t0

    chunk, gen_seconds, n_tokens = cfg["vllm"]["chunk"], 0.0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(jobs), chunk):
            part = jobs[s:s + chunk]
            t = time.perf_counter()
            outs = llm.generate(args.system, [j["prompt"] for j in part], [j["image"] for j in part], args.max_tokens)
            gen_seconds += time.perf_counter() - t
            for j, (text, n) in zip(part, outs):
                f.write(json.dumps({"k": j["k"], "v": text, "tokens": n}, ensure_ascii=False) + "\n")
                n_tokens += n
            print(f"{s + len(part)}/{len(jobs)} done, {gen_seconds:.1f}s", flush=True)
    with open(args.meta, "w") as f:
        json.dump({"prompts": len(jobs), "tokens": n_tokens, "load_seconds": load_seconds,
                   "generate_seconds": gen_seconds}, f, indent=1)


if __name__ == "__main__":
    main()
