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

"""Parallel bootstrap and model staging across the 8-host (32-chip) TPU v6e slice."""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

def resolve_default_gcs_uri() -> str:
    if os.environ.get("MODEL_URI"):
        return os.environ["MODEL_URI"]
    try:
        proj = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if proj:
            return f"gs://{proj}-deepseek-v6e/models/deepseek-ai/DeepSeek-V4.1-Flash"
    except Exception:
        pass
    return "gs://deepseek-v6e-models/models/deepseek-ai/DeepSeek-V4.1-Flash"


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INVENTORY = SCRIPT_DIR / "nodes_inventory.json"
DEFAULT_GCS_URI = resolve_default_gcs_uri()
HF_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
VLLM_TORCHTPU_BRANCH = "epic/dsv41"


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


def load_nodes(inventory_path: Path) -> list[dict]:
    with open(inventory_path) as f:
        return json.load(f)


def run_ssh(node: dict, script: str, timeout: int = 3600) -> tuple[int, int, str, str]:
    rank = node["rank"]
    nat = node["nat"]
    cmd = ["ssh", *SSH_OPTS, f"{USER}@{nat}", "bash -s"]
    proc = subprocess.run(
        cmd,
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return rank, proc.returncode, proc.stdout, proc.stderr


BOOTSTRAP_SCRIPT_TEMPLATE = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[Host $(hostname)] Step 1: Remounting /dev/shm to 256G..."
sudo mount -o remount,size=256G /dev/shm
df -h /dev/shm

echo "[Host $(hostname)] Step 2: Installing system build dependencies..."
for attempt in 1 2 3; do
  if sudo apt-get update -qq && sudo apt-get install -y -qq \
      curl git build-essential cmake ninja-build libgoogle-perftools4 \
      libibverbs1 jq psmisc lsof pciutils; then
    break
  fi
  sleep 5
done

echo "[Host $(hostname)] Step 3: Installing uv and creating Python 3.12 environment..."
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -d "$HOME/vllm_tpu_env" ]]; then
  uv venv --python 3.12 "$HOME/vllm_tpu_env"
fi
source "$HOME/vllm_tpu_env/bin/activate"

echo "[Host $(hostname)] Step 4: Cloning vllm-project/vllm-torchtpu (branch: __BRANCH__)..."
if [[ ! -d "$HOME/vllm-torchtpu/.git" ]]; then
  git clone --branch "__BRANCH__" --single-branch https://github.com/vllm-project/vllm-torchtpu.git "$HOME/vllm-torchtpu"
else
  cd "$HOME/vllm-torchtpu"
  git fetch origin "__BRANCH__"
  git checkout -f FETCH_HEAD
fi

cd "$HOME/vllm-torchtpu"
export CC=gcc CXX=g++ VLLM_TARGET_DEVICE=empty

echo "[Host $(hostname)] Step 5: Installing vllm-torchtpu dependencies via uv..."
uv pip install \
  --python "$HOME/vllm_tpu_env/bin/python" \
  --index https://us-python.pkg.dev/ml-oss-artifacts-published/torch-tpu-release-virtual-registry/simple/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  -r pyproject.toml \
  --extra benchmarking

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
  uv pip install --python "$HOME/vllm_tpu_env/bin/python" --no-deps -e .

echo "[Host $(hostname)] Step 6: Verifying TPU v6e hardware & Python imports..."
TPU_SKIP_MDS_QUERY=1 python -c "
import torch, torch_tpu, jax, vllm, vllm_torchtpu
print('PyTorch:', torch.__version__, 'JAX:', jax.__version__, 'vLLM:', vllm.__version__)
print('Local TPU devices visible to torch.accelerator:', torch.accelerator.device_count())
"
echo "[Host $(hostname)] Bootstrap COMPLETE."
"""


STAGE_SHARD_SCRIPT_TEMPLATE = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
source "$HOME/vllm_tpu_env/bin/activate"

NODE_RANK="__NODE_RANK__"
NUM_NODES="__NUM_NODES__"
HF_REPO="__HF_REPO__"
GCS_URI="__GCS_URI__"
LOCAL_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash"
mkdir -p "$LOCAL_MODEL_DIR"

python - "$NODE_RANK" "$NUM_NODES" "$HF_REPO" "$GCS_URI" "$LOCAL_MODEL_DIR" <<'PY'
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi

node_rank = int(sys.argv[1])
num_nodes = int(sys.argv[2])
repo = sys.argv[3]
gcs_uri = sys.argv[4].rstrip("/")
local_dir = Path(sys.argv[5])

info = HfApi().model_info(repo, files_metadata=True)
revision = info.sha

# Check existing objects in GCS
existing = {}
res = subprocess.run(
    ["gcloud", "storage", "ls", "-l", f"{gcs_uri}/**"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
if res.returncode == 0:
    prefix = gcs_uri + "/"
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2].startswith(prefix):
            rel = parts[2][len(prefix):]
            existing[rel] = int(parts[0])

all_files = [(s.rfilename, s.size) for s in info.siblings if s.size is not None]
my_files = [
    (fname, fsize)
    for idx, (fname, fsize) in enumerate(all_files)
    if idx % num_nodes == node_rank and existing.get(fname) != fsize
]

print(f"[Node {node_rank}] Total files: {len(all_files)}, assigned missing files: {len(my_files)}")

def copy_one(item):
    fname, fsize = item
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
    dest = f"{gcs_uri}/{fname}"
    for attempt in range(1, 4):
        cmd = f"curl -fsSL --retry 3 '{url}' | gcloud storage cp --quiet - '{dest}'"
        r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd])
        if r.returncode == 0:
            print(f"[Node {node_rank}] Staged {fname} ({fsize / (1024**2):.1f} MiB)")
            return True
    raise RuntimeError(f"Failed to stage {fname}")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(copy_one, my_files))

# Also download non-weight metadata/tokenizer files locally to ~/models/DeepSeek-V4.1-Flash
non_weights = [
    fname for fname, _ in all_files
    if not fname.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf"))
]
for fname in non_weights:
    target = local_dir / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
        subprocess.run(["curl", "-fsSL", "--retry", "3", url, "-o", str(target)], check=True)
print(f"[Node {node_rank}] Local tokenizer/config ready in {local_dir}")
PY
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--gcs-uri", default=DEFAULT_GCS_URI)
    parser.add_argument("--branch", default=VLLM_TORCHTPU_BRANCH)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--stage-weights", action="store_true", help="Stage model weights to GCS (enabled by default)")
    args = parser.parse_args()

    nodes = load_nodes(args.inventory)
    print(f"Loaded {len(nodes)} TPU v6e hosts from {args.inventory}")

    if not args.skip_bootstrap:
        print(f"\n=== Phase 1: Bootstrapping {len(nodes)} TPU v6e hosts in parallel ===")
        t0 = time.time()
        script = BOOTSTRAP_SCRIPT_TEMPLATE.replace("__BRANCH__", args.branch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
            futures = {ex.submit(run_ssh, n, script): n for n in nodes}
            for fut in concurrent.futures.as_completed(futures):
                rank, code, out, err = fut.result()
                print(f"\n--- [Node {rank}] Bootstrap exit={code} ---")
                print("\n".join(out.splitlines()[-15:]))
                if code != 0:
                    print(f"[Node {rank} STDERR]:\n{err}", file=sys.stderr)
                    raise RuntimeError(f"Bootstrap failed on node {rank}")
        print(f"=== Phase 1 completed in {time.time() - t0:.1f}s ===")

    if not args.skip_stage:
        print(f"\n=== Phase 2: Distributed staging of {HF_REPO} -> {args.gcs_uri} across {len(nodes)} hosts ===")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
            futures = {}
            for n in nodes:
                s = (
                    STAGE_SHARD_SCRIPT_TEMPLATE
                    .replace("__NODE_RANK__", str(n["rank"]))
                    .replace("__NUM_NODES__", str(len(nodes)))
                    .replace("__HF_REPO__", HF_REPO)
                    .replace("__GCS_URI__", args.gcs_uri)
                )
                futures[ex.submit(run_ssh, n, s)] = n
            for fut in concurrent.futures.as_completed(futures):
                rank, code, out, err = fut.result()
                print(f"\n--- [Node {rank}] Model staging exit={code} ---")
                print(out.strip())
                if code != 0:
                    print(f"[Node {rank} STDERR]:\n{err}", file=sys.stderr)
                    raise RuntimeError(f"Model staging failed on node {rank}")
        print(f"=== Phase 2 completed in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
