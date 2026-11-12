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

"""Functional verification suite for DeepSeek-V4.1-Flash on Multi-Host TPU v6e."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"


def http_get(url: str, timeout: int = 30) -> dict | str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode()
        try:
            return json.loads(body)
        except Exception:
            return body


def chat_completion(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    stream: bool = True,
    timeout: int = 300,
) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_token_t = None
    chunks = []
    prompt_tokens = 0
    completion_tokens = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            data = json.loads(resp.read().decode())
            t_end = time.perf_counter()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "text": text,
                "ttft_ms": (t_end - t0) * 1000.0,
                "e2e_ms": (t_end - t0) * 1000.0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            if "usage" in obj and obj["usage"]:
                prompt_tokens = obj["usage"].get("prompt_tokens", prompt_tokens)
                completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    chunks.append(content)
                    completion_tokens += 1

    t_end = time.perf_counter()
    if first_token_t is None:
        first_token_t = t_end
    text = "".join(chunks)
    return {
        "text": text,
        "ttft_ms": (first_token_t - t0) * 1000.0,
        "e2e_ms": (t_end - t0) * 1000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tps": max(completion_tokens - 1, 1) / max(t_end - first_token_t, 1e-6),
    }


def run_functional_suite(base_url: str, out_path: Path) -> None:
    print(f"Checking /v1/models on {base_url}...")
    models = http_get(f"{base_url}/v1/models")
    print("Models endpoint response:", json.dumps(models, indent=2))

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_NAME,
        "models_endpoint": models,
        "tests": [],
    }

    # Test 1: Deterministic Math & Systems Calculation
    print("\n[Test 1/5] Deterministic TPU Memory & Topology Calculation...")
    t1_msgs = [
        {"role": "system", "content": "You are a precise systems engineering assistant. Reply concisely."},
        {
            "role": "user",
            "content": (
                "A Cloud TPU v6e-32 slice consists of 8 ct6e-standard-4t hosts with 4 chips per host "
                "and 32 GB HBM per chip. If DeepSeek-V4.1-Flash has 384 routed experts sharded evenly "
                "across all 32 chips via Expert Parallelism (EP=32), how many routed experts reside on "
                "each TPU v6e chip, and what is the total HBM capacity of the 32-chip slice in GB?"
            ),
        },
    ]
    r1 = chat_completion(base_url, t1_msgs, max_tokens=192)
    passed_1 = ("12" in r1["text"]) and ("1024" in r1["text"] or "1,024" in r1["text"])
    print(f"  Passed={passed_1} | TTFT={r1['ttft_ms']:.1f}ms | E2E={r1['e2e_ms']:.1f}ms | Output: {r1['text'][:180]!r}")
    results["tests"].append({"name": "tpu_math_reasoning", "passed": passed_1, **r1})

    # Test 2: Agentic Tool-Use / Structured Function Calling
    print("\n[Test 2/5] Agentic Assistant Tool-Calling (JSON Schema)...")
    t2_msgs = [
        {
            "role": "system",
            "content": (
                "You are an autonomous coding agent. You have access to the following tools:\n"
                "1. `read_file(path: str, start_line: int, end_line: int)`\n"
                "2. `edit_file(path: str, target: str, replacement: str)`\n"
                "3. `run_command(command: str, timeout_ms: int)`\n"
                "Respond ONLY with a single valid JSON object containing keys `tool` and `arguments`."
            ),
        },
        {
            "role": "user",
            "content": (
                "Inspect lines 100 to 200 of `/workspace/vllm-torchtpu/src/vllm_torchtpu/platforms/tpu_platform.py` "
                "to verify the 2D torus topology map."
            ),
        },
    ]
    r2 = chat_completion(base_url, t2_msgs, max_tokens=192)
    passed_2 = ("read_file" in r2["text"]) and ("tpu_platform.py" in r2["text"])
    print(f"  Passed={passed_2} | TTFT={r2['ttft_ms']:.1f}ms | E2E={r2['e2e_ms']:.1f}ms | Output: {r2['text'][:180]!r}")
    results["tests"].append({"name": "agentic_tool_call", "passed": passed_2, **r2})

    # Test 3: Multi-Turn Coding Agent Debugging Trajectory
    print("\n[Test 3/5] Multi-Turn Coding Agent Traceback Debugging...")
    t3_msgs = [
        {"role": "system", "content": "You are an expert Python & distributed systems coding agent."},
        {"role": "user", "content": "Fix the bug in `prepare_dp_ports(base_port)` where peer hosts pick random ports."},
        {
            "role": "assistant",
            "content": '{"tool": "read_file", "arguments": {"path": "multihost_dp.py", "start_line": 1, "end_line": 40}}',
        },
        {
            "role": "user",
            "content": (
                "Tool output:\n```python\ndef prepare_dp_ports(rpc_port: int) -> list[int]:\n"
                "    import random\n"
                "    return [random.randint(20000, 30000) for _ in range(5)]\n```\n"
                "Provide the deterministic replacement function using offsets `rpc_port + 100 + i`."
            ),
        },
    ]
    r3 = chat_completion(base_url, t3_msgs, max_tokens=256)
    passed_3 = ("rpc_port + 100" in r3["text"]) or ("100 + i" in r3["text"])
    print(f"  Passed={passed_3} | TTFT={r3['ttft_ms']:.1f}ms | E2E={r3['e2e_ms']:.1f}ms | Output: {r3['text'][:180]!r}")
    results["tests"].append({"name": "multiturn_agent_debugging", "passed": passed_3, **r3})

    # Test 4: Long-Context Repository Analysis (~8K context)
    print("\n[Test 4/5] Long-Context Agentic Repository Analysis (~8K tokens)...")
    repo_modules = []
    for i in range(60):
        secret = "TOPOLOGY_SENTINEL_4X8_V6E32" if i == 42 else f"MODULE_CONST_{i}"
        repo_modules.append(
            f"# File: src/cluster/module_{i:02d}.py\n"
            f"class ClusterNode_{i:02d}:\n"
            f"    \"\"\"Handles shard routing and heartbeat verification for partition {i}.\"\"\"\n"
            f"    SENTINEL = \"{secret}\"\n"
            f"    def route_packet(self, token_batch: list[int], rank: int) -> dict:\n"
            f"        return {{'module': {i}, 'rank': rank, 'count': len(token_batch), 'sentinel': self.SENTINEL}}\n"
        )
    long_repo_prompt = (
        "Below is a multi-file repository snapshot for an agentic code audit:\n\n"
        + "\n".join(repo_modules)
        + "\nQuestion: What is the exact value of `SENTINEL` in `src/cluster/module_42.py`? Reply with the exact string."
    )
    r4 = chat_completion(
        base_url,
        [
            {"role": "system", "content": "You are a codebase analysis agent."},
            {"role": "user", "content": long_repo_prompt},
        ],
        max_tokens=128,
    )
    passed_4 = "TOPOLOGY_SENTINEL_4X8_V6E32" in r4["text"]
    print(f"  Passed={passed_4} | TTFT={r4['ttft_ms']:.1f}ms | E2E={r4['e2e_ms']:.1f}ms | Output: {r4['text'][:180]!r}")
    results["tests"].append({"name": "long_context_repo_needle", "passed": passed_4, **r4})

    # Test 5: Concurrent 8-Agent Burst across DP=32 / EP=32 Slice
    print("\n[Test 5/5] Concurrent 8-Agent Burst across Multi-Host DP=32 / EP=32...")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(
                chat_completion,
                base_url,
                [
                    {"role": "system", "content": "You are a concise coding assistant."},
                    {"role": "user", "content": f"Write a one-line Python lambda that multiplies x by {k + 2}."},
                ],
                128,
            )
            for k in range(8)
        ]
        burst_res = [f.result() for f in futs]
    burst_wall_ms = (time.perf_counter() - t0) * 1000.0
    passed_5 = all(len(x["text"].strip()) > 0 for x in burst_res)
    print(f"  Passed={passed_5} | 8 concurrent requests completed in {burst_wall_ms:.1f}ms")
    results["tests"].append({
        "name": "concurrent_8_agent_burst",
        "passed": passed_5,
        "burst_wall_ms": burst_wall_ms,
        "individual_ttft_ms": [round(x["ttft_ms"], 2) for x in burst_res],
        "individual_e2e_ms": [round(x["e2e_ms"], 2) for x in burst_res],
    })

    results["all_passed"] = all(t["passed"] for t in results["tests"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved functional verification report to {out_path} (all_passed={results['all_passed']})")
    if not results["all_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("results/functional_verification.json"))
    args = parser.parse_args()
    run_functional_suite(args.base_url, args.out)


if __name__ == "__main__":
    main()
