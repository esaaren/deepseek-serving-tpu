#!/usr/bin/env bash
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

# Per-host vLLM-TorchTPU launcher for DeepSeek-V4.1-Flash on Multi-Host TPU v6e.
#
# Runs on each ct6e-standard-4t host (4 TPU v6e chips per host).
# Host 0 runs the OpenAI API server + local DP ranks [0..3].
# Hosts 1..N-1 run in --headless mode owning local DP ranks [host_rank*4 .. (host_rank+1)*4 - 1].
set -Eeuo pipefail
ulimit -c 0

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
source "$HOME/vllm_tpu_env/bin/activate"

HOST_RANK="${HOST_RANK:?HOST_RANK is required (0..NUM_HOSTS-1)}"
NUM_HOSTS="${NUM_HOSTS:-8}"
HEAD_IP="${HEAD_IP:?HEAD_IP (internal IP of host 0) is required}"

LOCAL_DP="${LOCAL_DP:-4}"
TP="${TP:-1}"
DP="${DP:-$((NUM_HOSTS * LOCAL_DP))}"

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || echo 'deepseek-v6e')}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/DeepSeek-V4.1-Flash}"
MODEL_URI="${MODEL_URI:-gs://${PROJECT}-deepseek-v6e/models/deepseek-ai/DeepSeek-V4.1-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"

EXECUTION_MODE="${EXECUTION_MODE:-ced}"
VLLM_TPU_GLOO_EP="${VLLM_TPU_GLOO_EP:-1}"
if [[ "${VLLM_TPU_GLOO_EP}" == "1" ]]; then
  MOE_BACKEND="${MOE_BACKEND:-gmm}"
else
  MOE_BACKEND="${MOE_BACKEND:-fused_ep_w4a8}"
fi
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-${KV_CACHE_DTYPE}}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
COMPILE_SIZES="${COMPILE_SIZES:-[64,256,512,2048]}"
ENGRAM_HBM="${ENGRAM_HBM:-off}"

# Use /dev/shm for compiler scratch and Tier-3 compilation cache so disk never fills
SCRATCH_ROOT="/dev/shm/dsv41-serve"
mkdir -p "$SCRATCH_ROOT/tmp" "$SCRATCH_ROOT/tier3_cache"
export TMPDIR="$SCRATCH_ROOT/tmp"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$SCRATCH_ROOT/tier3_cache"
export VLLM_TPU_GLOO_EP="${VLLM_TPU_GLOO_EP}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Clear any stale single-host or Megascale variables so native mp multihost DP owns bootstrap
unset MEGASCALE_COORDINATOR_ADDRESS MEGASCALE_NUM_SLICES MEGASCALE_PORT MEGASCALE_SLICE_ID
unset TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT TPU_WORKER_HOSTNAMES TORCH_TPU_SLICEBUILDER_ADDRESSES

export OMP_NUM_THREADS=1
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_TARGET_DEVICE=tpu
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export RUNAI_STREAMER_MEMORY_LIMIT=34359738368
export TPU_SHARDED_LOAD_SYNC_EVERY=512
export TPU_RECLAIM_COMPILE_MEMORY=0
export LD_PRELOAD=""

export TPU_MULTIHOST_BACKEND="mp"
export TPU_ACCELERATOR_TYPE="v6e-4"
export TPU_SKIP_MDS_QUERY="1"
export TORCH_TPU_DP_MASTER_ADDR="${HEAD_IP}"
export TORCH_TPU_DP_MASTER_PORT="19700"
export MASTER_ADDR="${HEAD_IP}"
export MASTER_PORT="19700"
export TORCH_TPU_BASE_PORT="19000"
export TPU_DEEPSEEK_V41_ENGRAM_HBM="${ENGRAM_HBM}"

export LIBTPU_INIT_ARGS="--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_tpu_dvfs_p_state=7 --xla_tpu_impure_enable_large_2nd_minor_layout=false --xla_tpu_impure_enable_large_2nd_minor_entry_layout=false --xla_tpu_impure_use_large_2nd_minor_layout_for_x16=false"
if [[ "${MOE_FUSED_EP_PREPROCESS_BACKEND:-native}" == "topology" ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS} --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
fi

ENGRAM_CONFIG='{"cpu_offload":true,"dp_shared_memory":false}'
ADDITIONAL_CONFIG="{\"deepseek_v41\":{\"execution\":\"${EXECUTION_MODE}\",\"moe\":\"${MOE_BACKEND}\"}}"

SERVE_ARGS=(
  "$MODEL_URI"
  --tokenizer "$MODEL_PATH"
  --load-format runai_streamer
  --served-model-name "$SERVED_MODEL_NAME"
  --host "0.0.0.0"
  --port "${PORT:-8000}"
  --tensor-parallel-size "$TP"
  --data-parallel-size "$DP"
  --data-parallel-size-local "$LOCAL_DP"
  --data-parallel-address "$HEAD_IP"
  --data-parallel-rpc-port 19500
  --enable-expert-parallel
  --enable-ep-weight-filter
  --language-model-only
  --async-scheduling
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-config "{\"indexer_kv_dtype\":\"${INDEXER_KV_DTYPE}\"}"
  --engram-config "$ENGRAM_CONFIG"
  --additional-config "$ADDITIONAL_CONFIG"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --compilation-config "{\"compile_sizes\":${COMPILE_SIZES}}"
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  SERVE_ARGS+=(--enable-prefix-caching --swa-bounded-replay)
else
  SERVE_ARGS+=(--no-enable-prefix-caching)
fi

if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  SERVE_ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"${MODEL_URI}\"}")
fi

if [[ "${HOST_RANK}" -eq 0 ]]; then
  SERVE_ARGS+=(--api-server-count 1)
else
  START_RANK=$((HOST_RANK * LOCAL_DP))
  SERVE_ARGS+=(--headless --data-parallel-start-rank "${START_RANK}")
fi

echo "=================================================================="
echo " Launching DeepSeek-V4.1-Flash on Host Rank ${HOST_RANK}/${NUM_HOSTS}"
echo " Head IP: ${HEAD_IP} | TP=${TP} | DP=${DP} (local=${LOCAL_DP}) | EP=True"
echo " Execution=${EXECUTION_MODE} | MoE=${MOE_BACKEND} | KV=${KV_CACHE_DTYPE}"
echo " MaxModelLen=${MAX_MODEL_LEN} | MaxBatchedTokens=${MAX_NUM_BATCHED_TOKENS} | MaxSeqs=${MAX_NUM_SEQS}"
echo "=================================================================="

exec vllm serve "${SERVE_ARGS[@]}" "$@"
