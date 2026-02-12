import os
import json
import time
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

RAW_OUT = Path("outputs/raw.jsonl")
DATA_PATH = Path("data/seed_trap_set.json")

def load_seed_traps():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def call_llm(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    # Keep it simple: single-turn chat
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a careful problem solver. Output your reasoning, then a final answer."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""

def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)

    seed_traps = load_seed_traps()
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    temperature = 0.7
    samples_per_prompt = 1  # 10–20 is good
    sleep_s = 0.3

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(RAW_OUT, "a", encoding="utf-8") as fout:
        for item in seed_traps:
            pid = item["id"]
            for prompt_type, prompts in item["prompt_family"].items():
                for p_idx, prompt in enumerate(prompts):
                    for s in range(samples_per_prompt):
                        # small jitter to avoid burst rate spikes
                        time.sleep(sleep_s + random.random() * 0.2)
                        try:
                            output = call_llm(client, model, prompt, temperature)
                        except Exception as e:
                            output = f"__ERROR__: {repr(e)}"

                        record = {
                            "run_tag": run_tag,
                            "model": model,
                            "temperature": temperature,
                            "samples_per_prompt": samples_per_prompt,
                            "problem_id": pid,
                            "prompt_type": prompt_type,
                            "prompt_index": p_idx,
                            "sample_index": s,
                            "prompt": prompt,
                            "output_text": output,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()

    print(f"Done. Wrote: {RAW_OUT}")

if __name__ == "__main__":
    main()
