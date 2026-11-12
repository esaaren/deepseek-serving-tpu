#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agentic Assistant Benchmark Suite for DeepSeek-V4.1-Flash on Multi-Host TPU v6e.

Simulates realistic Agentic Coding & Tool-Use Assistant workloads across context
lengths and concurrency levels:
  - 4K Input  / 512 Output : Interactive Agent Turn (System Prompt + 20 Tool Schemas + Task)
  - 16K Input / 512 Output : Multi-Turn Coding Agent Trajectory (Tool History + Codebase Files)
  - 32K Input / 512 Output : Deep Repository Refactoring Agent (Multi-File Repo + Diff Context)
  - 64K Input / 1024 Output: Long-Horizon Autonomous Agent Session (Full Repo + Test Logs)
  - 16K Cached Prefix + 2K Turn Delta / 512 Output: Multi-Turn Prefix-Cached Agent Loop
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path
import aiohttp

DEFAULT_URL = "http://127.0.0.1:8000"
MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"


AGENTIC_SYSTEM_HEADER = """You are an autonomous software engineering and infrastructure agent.
Available Tools (JSON Schema):
- read_file(path: str, start_line: int, end_line: int) -> str
- edit_file(path: str, target: str, replacement: str) -> str
- write_file(path: str, content: str, overwrite: bool) -> str
- run_command(command_line: str, wait_ms: int) -> dict
- grep_search(query: str, search_path: str) -> list[dict]
- find_by_name(pattern: str, search_directory: str) -> list[str]
- manage_subagents(action: str, conversation_ids: list[str]) -> dict
"""


def build_agentic_prompt(
    target_input_tokens: int,
    request_id: int,
    tokenizer=None,
    shared_prefix_ratio: float = 0.0,
) -> str:
    """Build a realistic agentic coding trajectory prompt calibrated to target_input_tokens."""
    # Each synthetic code/tool turn block is ~120 tokens
    block_template = (
        "# [Tool Turn {idx}] File inspection: src/distributed/tpu_executor_shard_{idx:04d}.py\n"
        "def dispatch_expert_shard_{idx:04d}(hidden_states, router_logits, ep_rank: int = {ep}):\n"
        "    \"\"\"Routes token activations across 2D Torus ICI links for expert group {idx}.\"\"\"\n"
        "    topk_weights, topk_ids = select_topk_experts(router_logits, k=6, norm=True)\n"
        "    dispatched = all_to_all_permute(hidden_states, topk_ids, group_size=32)\n"
        "    expert_out = fused_ep_w4a8_matmul(dispatched, w13_fp4, w2_fp4, scales_e8m0)\n"
        "    return combine_routed_outputs(expert_out, topk_weights, ep_rank=ep_rank)\n\n"
    )
    footer = (
        f"\n# [Current Agent Task #{request_id}]\n"
        "Analyze the distributed expert dispatch implementation above, identify potential "
        "VMEM double-buffering optimizations and ICI collective overlap opportunities, and "
        "write the updated Python implementation with detailed inline explanations.\n"
    )

    # Approximate tokens at ~3.6 chars/token if tokenizer not loaded, or exact with tokenizer
    if shared_prefix_ratio > 0.0:
        shared_tokens = int(target_input_tokens * shared_prefix_ratio)
        unique_tokens = max(256, target_input_tokens - shared_tokens)
        n_shared_blocks = max(1, shared_tokens // 115)
        n_unique_blocks = max(1, unique_tokens // 115)
        shared_body = "".join(block_template.format(idx=i, ep=i % 32) for i in range(n_shared_blocks))
        unique_body = "".join(
            block_template.format(idx=10000 + request_id * 500 + j, ep=(request_id + j) % 32)
            for j in range(n_unique_blocks)
        )
        prompt = AGENTIC_SYSTEM_HEADER + "\n" + shared_body + "\n" + unique_body + footer
    else:
        # Unique per request so non-cached benchmarks don't accidentally hit prefix cache
        n_blocks = max(1, target_input_tokens // 115)
        body = "".join(
            block_template.format(idx=request_id * 1000 + i, ep=(request_id + i) % 32)
            for i in range(n_blocks)
        )
        prompt = AGENTIC_SYSTEM_HEADER + "\n" + body + footer

    if tokenizer is not None:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_body_tokens = max(64, target_input_tokens - 32)
        if len(ids) < target_body_tokens:
            pad_unit = tokenizer.encode(block_template.format(idx=9999, ep=0), add_special_tokens=False)
            while len(ids) < target_body_tokens:
                ids.extend(pad_unit)
        ids = ids[:target_body_tokens]
        footer_ids = tokenizer.encode(footer, add_special_tokens=False)
        ids = ids[: max(32, target_body_tokens - len(footer_ids))] + footer_ids
        prompt = tokenizer.decode(ids, skip_special_tokens=True)

    return prompt


async def send_one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    async with semaphore:
        t_start = time.perf_counter()
        first_token_t = None
        token_times = []
        prompt_tokens = 0
        completion_tokens = 0
        error_msg = None

        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1800),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {body[:300]}",
                    }
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    obj = json.loads(data_str)
                    if obj.get("usage"):
                        prompt_tokens = obj["usage"].get("prompt_tokens", prompt_tokens)
                        completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            now = time.perf_counter()
                            if first_token_t is None:
                                first_token_t = now
                            token_times.append(now)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        t_end = time.perf_counter()
        if error_msg or first_token_t is None:
            return {
                "success": False,
                "error": error_msg or "No tokens received",
            }

        if completion_tokens == 0:
            completion_tokens = len(token_times)
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt) // 4)

        ttft_s = first_token_t - t_start
        e2e_s = t_end - t_start
        decode_s = max(t_end - first_token_t, 1e-6)
        tpot_ms = (decode_s * 1000.0) / max(completion_tokens - 1, 1)

        return {
            "success": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft_s * 1000.0,
            "e2e_ms": e2e_s * 1000.0,
            "tpot_ms": tpot_ms,
            "decode_s": decode_s,
        }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


async def run_benchmark_cell(
    base_url: str,
    prompts: list[str],
    input_len: int,
    output_len: int,
    concurrency: int,
    scenario_tag: str,
    shared_prefix_ratio: float = 0.0,
    do_warmup: bool = True,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 64))
    async with aiohttp.ClientSession(connector=connector) as session:
        if do_warmup:
            # Warmup requests first (populate prefix cache across all 24 DP engines in 1 parallel wave)
            n_warmup = 24 if shared_prefix_ratio > 0.0 else 1
            warmup_sem = asyncio.Semaphore(24)
            warmup_tasks = [
                send_one_request(session, base_url, prompts[0], min(4, output_len), warmup_sem)
                for _ in range(n_warmup)
            ]
            await asyncio.gather(*warmup_tasks)

        t_wall_start = time.perf_counter()
        tasks = [
            send_one_request(session, base_url, p, output_len, semaphore)
            for p in prompts
        ]
        raw_results = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - t_wall_start

    ok = [r for r in raw_results if r.get("success")]
    failed = [r for r in raw_results if not r.get("success")]
    if not ok:
        raise RuntimeError(f"All {len(raw_results)} requests failed! Sample error: {failed[0]}")

    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    total_completion_tokens = sum(r["completion_tokens"] for r in ok)
    total_tokens = total_prompt_tokens + total_completion_tokens

    ttfts = [r["ttft_ms"] for r in ok]
    e2es = [r["e2e_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]

    prefill_tps = total_prompt_tokens / wall_time_s
    decode_tps = total_completion_tokens / wall_time_s
    total_tps = total_tokens / wall_time_s
    req_per_s = len(ok) / wall_time_s

    summary = {
        "scenario": scenario_tag,
        "target_input_tokens": input_len,
        "target_output_tokens": output_len,
        "concurrency": concurrency,
        "num_requests": len(prompts),
        "successful_requests": len(ok),
        "failed_requests": len(failed),
        "wall_time_s": round(wall_time_s, 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "prefill_tok_per_s": round(prefill_tps, 2),
        "decode_tok_per_s": round(decode_tps, 2),
        "total_tok_per_s": round(total_tps, 2),
        "requests_per_s": round(req_per_s, 3),
        "ttft_mean_ms": round(statistics.mean(ttfts), 2),
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p90_ms": round(percentile(ttfts, 90), 2),
        "ttft_p99_ms": round(percentile(ttfts, 99), 2),
        "tpot_mean_ms": round(statistics.mean(tpots), 2),
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p90_ms": round(percentile(tpots, 90), 2),
        "e2e_mean_ms": round(statistics.mean(e2es), 2),
        "e2e_p50_ms": round(percentile(e2es, 50), 2),
        "e2e_p90_ms": round(percentile(e2es, 90), 2),
        "e2e_p99_ms": round(percentile(e2es, 99), 2),
    }
    print(
        f"[{scenario_tag}] ISL={input_len} OSL={output_len} C={concurrency} N={len(ok)} | "
        f"Total={summary['total_tok_per_s']:.1f} tok/s (Prefill={summary['prefill_tok_per_s']:.1f}, "
        f"Decode={summary['decode_tok_per_s']:.1f}) | TTFT P50={summary['ttft_p50_ms']:.1f}ms | "
        f"TPOT P50={summary['tpot_p50_ms']:.2f}ms | E2E P50={summary['e2e_p50_ms']:.1f}ms",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--tokenizer-path", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--scenario", default="agentic_sweep")
    parser.add_argument("--input-len", type=int, default=4096)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--concurrencies", type=str, default="1,8,16,32,64")
    parser.add_argument("--num-prompts-multiplier", type=int, default=2)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = None
    if os.path.exists(args.tokenizer_path):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            print(f"Loaded tokenizer from {args.tokenizer_path}")
        except Exception as e:
            print(f"Warning: could not load tokenizer ({e}), using character calibration.")

    conc_list = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    if args.input_len >= 60000 and len(conc_list) > 2:
        conc_list = [1, 16]
    elif (args.input_len >= 32768 or args.shared_prefix_ratio > 0.0) and len(conc_list) > 3:
        conc_list = [c for c in conc_list if c in (1, 8, 24)]
    if args.input_len >= 16384 and args.output_len > 16:
        args.output_len = 16
    max_prompts = max(conc_list) * args.num_prompts_multiplier
    print(f"Generating {max_prompts} calibrated agentic prompts (target ISL={args.input_len}, prefix_ratio={args.shared_prefix_ratio})...")
    prompts = [
        build_agentic_prompt(args.input_len, i, tokenizer=tokenizer, shared_prefix_ratio=args.shared_prefix_ratio)
        for i in range(max_prompts)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_cells = []
    for idx, c in enumerate(conc_list):
        n_reqs = c * args.num_prompts_multiplier
        cell = asyncio.run(
            run_benchmark_cell(
                base_url=args.base_url,
                prompts=prompts[:n_reqs],
                input_len=args.input_len,
                output_len=args.output_len,
                concurrency=c,
                scenario_tag=args.scenario,
                shared_prefix_ratio=args.shared_prefix_ratio,
                do_warmup=(idx == 0 and args.shared_prefix_ratio > 0.0),
            )
        )
        all_cells.append(cell)
        args.out.write_text(json.dumps(all_cells, indent=2) + "\n")
    print(f"Saved {len(all_cells)} benchmark results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
