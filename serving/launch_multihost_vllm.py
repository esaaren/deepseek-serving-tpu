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

"""Cluster-wide launcher and health monitor for DeepSeek-V4.1-Flash on Multi-Host TPU v6e."""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_INVENTORY = REPO_DIR / "infra" / "nodes_inventory.json"


def resolve_ssh_user() -> str:
    if os.environ.get("TPU_SSH_USER"):
        return os.environ["TPU_SSH_USER"]
    try:
        account = subprocess.check_output(
            ["gcloud", "config", "get-value", "account"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if account and "@" in account:
            return re.sub(r"[^a-zA-Z0-9]", "_", account)
    except Exception:
        pass
    return os.environ.get("USER", "ubuntu")


SSH_KEY = os.environ.get("TPU_SSH_KEY", os.path.expanduser("~/.ssh/google_compute_engine"))
SSH_OPTS = [
    "-i", SSH_KEY,
    "-o", "ProxyCommand=none",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
]
USER = resolve_ssh_user()


def ssh_run(node: dict, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, f"{USER}@{node['nat']}", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def stop_cluster(nodes: list[dict]) -> None:
    print(f"Stopping any running vLLM processes across {len(nodes)} TPU v6e hosts...")
    kill_cmd = (
        "sudo killall -9 vllm python3 python 2>/dev/null || true; "
        "sudo fuser -k /dev/vfio/* 2>/dev/null || true; "
        "rm -rf /dev/shm/dsv41-serve /dev/shm/vllm_engram_* 2>/dev/null || true"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        list(ex.map(lambda n: ssh_run(n, kill_cmd, timeout=60), nodes))
    time.sleep(2)


def start_cluster(nodes: list[dict], env_overrides: dict[str, str], log_prefix: str = "vllm_serve") -> None:
    head_ip = nodes[0]["internal"]
    num_hosts = len(nodes)

    def _start_node(node: dict) -> None:
        rank = node["rank"]
        env_vars = {
            "HOST_RANK": str(rank),
            "NUM_HOSTS": str(num_hosts),
            "HEAD_IP": head_ip,
            **env_overrides,
        }
        exports = " ".join(f"{k}='{v}'" for k, v in env_vars.items())
        log_file = f"$HOME/{log_prefix}_rank{rank}.log"
        cmd = (
            f"nohup env {exports} bash $HOME/deepseek-serving-tpu/serving/serve_node.sh "
            f"> {log_file} 2>&1 < /dev/null & echo $!"
        )
        res = ssh_run(node, cmd, timeout=30)
        print(f"  [Node {rank} ({node['internal']})] Launched PID={res.stdout.strip()} -> {log_file}")

    print(f"Starting DeepSeek-V4.1-Flash across {num_hosts} hosts (Head IP: {head_ip})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_hosts) as ex:
        list(ex.map(_start_node, nodes))


def wait_until_ready(nodes: list[dict], log_prefix: str = "vllm_serve", timeout_s: int = 5400) -> float:
    head = nodes[0]
    t0 = time.time()
    print(f"Waiting for http://127.0.0.1:8000/health on Head Node 0 ({head['nat']})...")
    while time.time() - t0 < timeout_s:
        elapsed = time.time() - t0
        # Check health endpoint on head
        try:
            res = ssh_run(head, "curl -sf --max-time 5 http://127.0.0.1:8000/health", timeout=30)
            if res.returncode == 0:
                print(f"\n[READY] vLLM multi-host server is HEALTHY after {elapsed:.1f}s!")
                return elapsed
        except subprocess.TimeoutExpired:
            pass

        # Check if processes are still alive and print tail of rank 0 log
        try:
            tail_res = ssh_run(
                head,
                f"pgrep -f '[v]llm serve' >/dev/null && tail -n 6 $HOME/{log_prefix}_rank0.log || echo 'PROCESS_EXITED'",
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            time.sleep(5)
            continue
        out = tail_res.stdout.strip()
        if (
            "PROCESS_EXITED" in out
            or "Engine core initialization failed" in out
            or "WorkerProc failed" in out
            or "ValidationError" in out
            or "Traceback (most recent call last)" in out
        ):
            print(f"\n[ERROR] Server process failed on Node 0 after {elapsed:.1f}s!", file=sys.stderr)
            for n in nodes[:2]:
                err_log = ssh_run(n, f"tail -n 50 $HOME/{log_prefix}_rank{n['rank']}.log", timeout=20)
                print(f"--- Node {n['rank']} log tail ---\n{err_log.stdout}", file=sys.stderr)
            raise RuntimeError("vLLM multi-host server failed during startup")

        last_line = out.splitlines()[-1] if out.splitlines() else "..."
        print(f"  [{elapsed:6.1f}s] {last_line[:115]}")
        time.sleep(15)

    raise TimeoutError(f"Server did not become healthy within {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--action", choices=["start", "stop", "restart", "status", "wait"], default="restart")
    parser.add_argument("--execution-mode", choices=["ced", "full"], default="ced")
    parser.add_argument("--moe-backend", choices=["fused_ep_w4a8", "gmm"], default="gmm")
    parser.add_argument("--kv-cache-dtype", choices=["fp8", "auto"], default="fp8")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--enable-prefix-caching", type=int, choices=[0, 1], default=1)
    parser.add_argument("--enable-dspark", type=int, choices=[0, 1], default=0)
    parser.add_argument("--compile-sizes", type=str, default="[64,256,512,2048]")
    parser.add_argument("--log-prefix", type=str, default="vllm_serve")
    args = parser.parse_args()

    with open(args.inventory) as f:
        nodes = json.load(f)

    if args.action in ("stop", "restart"):
        stop_cluster(nodes)
        if args.action == "stop":
            return

    if args.action == "status":
        for n in nodes:
            r = ssh_run(n, f"tail -n 5 $HOME/{args.log_prefix}_rank{n['rank']}.log")
            print(f"=== Node {n['rank']} ===\n{r.stdout.strip()}")
        return

    if args.action == "wait":
        wait_until_ready(nodes, log_prefix=args.log_prefix)
        return

    env_overrides = {
        "EXECUTION_MODE": args.execution_mode,
        "MOE_BACKEND": args.moe_backend,
        "KV_CACHE_DTYPE": args.kv_cache_dtype,
        "MAX_MODEL_LEN": str(args.max_model_len),
        "MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens),
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "ENABLE_PREFIX_CACHING": str(args.enable_prefix_caching),
        "ENABLE_DSPARK": str(args.enable_dspark),
        "COMPILE_SIZES": args.compile_sizes,
    }
    start_cluster(nodes, env_overrides, log_prefix=args.log_prefix)
    wait_until_ready(nodes, log_prefix=args.log_prefix)


if __name__ == "__main__":
    main()
