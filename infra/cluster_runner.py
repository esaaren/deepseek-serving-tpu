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

"""Cluster-wide orchestrator for multi-host TPU v6e slice."""

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
DEFAULT_INVENTORY = SCRIPT_DIR / "nodes_inventory.json"


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


def load_nodes(inventory_path: Path = DEFAULT_INVENTORY) -> list[dict]:
    if not inventory_path.exists():
        return []
    with open(inventory_path) as f:
        return json.load(f)


def run_on_node(node: dict, cmd_str: str, timeout: int = 3600) -> tuple[int, int, str, str]:
    rank = node["rank"]
    nat = node["nat"]
    cmd = ["ssh", *SSH_OPTS, f"{USER}@{nat}", cmd_str]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return rank, res.returncode, res.stdout, res.stderr


def run_all(nodes: list[dict], cmd_str: str, timeout: int = 3600) -> bool:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        futures = {ex.submit(run_on_node, n, cmd_str, timeout): n for n in nodes}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda x: x[0])
    ok = True
    for rank, code, out, err in results:
        print(f"=== [Node {rank}] (exit={code}) ===")
        if out.strip():
            print(out.strip())
        if err.strip():
            print(err.strip(), file=sys.stderr)
        if code != 0:
            ok = False
    return ok


def sync_all(nodes: list[dict], local_dir: Path, remote_dir: str = "deepseek-serving-tpu") -> None:
    ssh_cmd = "ssh " + " ".join(SSH_OPTS)

    def _rsync(node: dict) -> tuple[int, int, str]:
        cmd = [
            "rsync", "-az", "--delete",
            "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
            "-e", ssh_cmd,
            str(local_dir).rstrip("/") + "/",
            f"{USER}@{node['nat']}:{remote_dir}/",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return node["rank"], res.returncode, res.stderr

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        for rank, code, err in ex.map(_rsync, nodes):
            if code != 0:
                raise RuntimeError(f"rsync failed on node {rank}: {err}")
    print(f"Synced {local_dir} -> ~/{remote_dir} across {len(nodes)} nodes in {time.time() - t0:.1f}s.")


def cleanup_all(nodes: list[dict]) -> None:
    kill_cmd = (
        "sudo killall -9 vllm python3 python 2>/dev/null || true; "
        "sudo fuser -k /dev/vfio/* 2>/dev/null || true; "
        "rm -rf /dev/shm/vllm_engram_* /dev/shm/dsv41-serve 2>/dev/null || true"
    )
    run_all(nodes, kill_cmd, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--nodes", type=str, default="", help="Comma-separated node ranks to run on")
    parser.add_argument("--sync", action="store_true", help="Sync repository to all nodes")
    parser.add_argument("--cleanup", action="store_true", help="Kill vLLM/TPU processes on all nodes")
    parser.add_argument("--cmd", type=str, help="Shell command to execute across all nodes")
    args = parser.parse_args()

    nodes = load_nodes(args.inventory)
    if args.nodes:
        selected = {int(x) for x in args.nodes.split(",") if x.strip()}
        nodes = [n for n in nodes if n["rank"] in selected]
    if not nodes:
        raise SystemExit(f"No nodes found in {args.inventory}")

    if args.cleanup:
        cleanup_all(nodes)
    if args.sync:
        sync_all(nodes, SCRIPT_DIR.parent)
    if args.cmd:
        if not run_all(nodes, args.cmd):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
