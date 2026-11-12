# TPU v6e Provisioning, Porting, and Benchmark Guide

This guide covers provisioning multi-host Cloud TPU v6e (Trillium) capacity on Google Cloud Compute Engine, applying the TPU v6e hardware adaptations for [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (`552B` total parameters) on [`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu) (`epic/dsv41`), and running the verification and benchmark suites.

---

## 1. Checking Quota & Spot / DWS Flex Capacity

Run [`infra/check_tpu_v6e_capacity.sh`](../infra/check_tpu_v6e_capacity.sh) to inspect regional `TPU_V6E` quota (`PREEMPTIBLE_TPU_V6E_CHIPS`, `TPU_V6E_CHIPS`) and verify `ct6e-standard-4t` machine-type availability across candidate zones:

```bash
bash infra/check_tpu_v6e_capacity.sh
```

### Capacity Calculation for `DeepSeek-V4.1-Flash` (552B)
- **Model Checkpoint Size:** `475.27 GiB` across 48 `.safetensors` shards (`384` routed MoE experts + `1` shared expert in MXFP4/INT4, `60` layers, `d_model = 5120`, MLA + Sliding-Window Attention + Engram N-gram memory).
- **Per-Chip HBM Budget on TPU v6e (`32 GiB` HBM2e per chip):**
  - **6 Hosts × `ct6e-standard-4t` = `24` TPU v6e chips (`768 GiB` HBM, `4,320 GiB` Host DRAM)** with `DP=24, EP=24, TP=1`:
    - Replicated attention/dense weights + `16` routed experts per chip (`384 / 24`): **`22.10 GiB` / chip**
    - FP8 KV cache + FP8 Sparse Indexer cache (`933,080` tokens / chip = **`22.39M` tokens cluster-wide**): **`5.05 GiB` / chip**
    - Compiled PjRt executables (`[64, 256, 512]` buckets) + runtime scratch: **`1.49 GiB` / chip**
    - Remaining free HBM headroom: **`3.11 GiB` / chip**
  - **8 Hosts × `ct6e-standard-4t` = `32` TPU v6e chips (`1,024 GiB` HBM, `5,760 GiB` Host DRAM)** with `DP=32, EP=32, TP=1`:
    - `12` routed experts per chip (`384 / 32`), reducing per-chip expert footprint by `25%` (`~19.8 GiB` weights/chip).

---

## 2. Provisioning TPU v6e Hosts (Spot or DWS Flex Start)

Do not use the legacy `gcloud compute tpus tpu-vm` (`tpu.googleapis.com`) API, which can fail with `403 SERVICE_PRODUCER_NOT_AUTHORIZED` on modern VPC setups. Provision `ct6e-standard-4t` Compute Engine VMs directly via [`infra/provision_tpu_v6e_mig.sh`](../infra/provision_tpu_v6e_mig.sh).

### Option A: Spot Provisioning (6 Hosts = 24 Chips or 8 Hosts = 32 Chips)
```bash
# 6x ct6e-standard-4t (24 TPU v6e chips) in asia-south1-c or us-west1-c
ZONE=asia-south1-c TARGET_SIZE=6 TOPOLOGY=none SPOT=true \
  BASE_NAME=dsv41-as1c bash infra/provision_tpu_v6e_mig.sh
```

### Option B: Dynamic Workload Scheduler (DWS) Flex Mode (`FLEX_START`)
When using DWS Flex capacity (e.g., `us-central1-a`, `southamerica-west1-a`, `us-east5-b`), set `SPOT=false` and `PROVISIONING_MODEL=FLEX_START`:
```bash
ZONE=us-central1-a TARGET_SIZE=8 TOPOLOGY=4x8 SPOT=false \
  PROVISIONING_MODEL=FLEX_START MAX_RUN_DURATION=6h \
  BASE_NAME=dsv41-flex bash infra/provision_tpu_v6e_mig.sh
```

The provisioning script writes the node inventory (ranks, internal IPs, and external NAT IPs) to [`infra/nodes_inventory.json`](../infra/nodes_inventory.json).

---

## 3. Bootstrapping & TPU v6e Hardware Adaptations

### 3.1 Host Environment & GCS Weight Staging
Run [`infra/cluster_setup.py`](../infra/cluster_setup.py) to install `vllm` + `vllm-torchtpu` (`epic/dsv41`) + `torch_tpu` (`0.0.50`) in `~/vllm_tpu_env` on every host and stream the `DeepSeek-V4.1-Flash` checkpoint into a regional GCS bucket (`gs://<project>-deepseek-v6e-<region>/models/deepseek-ai/DeepSeek-V4.1-Flash`) for parallel `runai_streamer` loading:

```bash
python3 infra/cluster_setup.py --inventory infra/nodes_inventory.json --stage-weights
```

### 3.2 Applying the TPU v6e (Trillium) Patch Set
The upstream `epic/dsv41` branch targets TPU v7x (`ChipVersion.TPU_7X`, `512 KB` SparseCore VMEM, hardware `float4_e2m1fn`). [`infra/apply_multihost_gloo_ep_patch.py`](../infra/apply_multihost_gloo_ep_patch.py) applies six deterministic adaptations for TPU v6e (`ChipVersion.TPU_V6E`, `256 KB` SparseCore VMEM, `32 GiB` HBM):

1. **W4A8 (`INT8 × INT4`) Grouped Matrix Multiply for MoE (`layers/core/fused_moe_gmm.py`, `mxfp4.py`):** Replaces TPU v7x's `jnp.float4_e2m1fn` weight packing with signed 4-bit integer (`jnp.int4`) block quantization packed into `uint8` pairs, executing natively on TPU v6e MXUs.
2. **SparseCore 256 KB VMEM Gather Tile Tuning (`worker/tpu_worker.py`):** Overrides `PAGED_MQA_GATHER_CONFIG[(8, 1)] = (32, 2)`, `PAGED_MQA_GATHER_CONFIG[(8, 2)] = (32, 2)`, and `PAGED_MQA_FRAGMENT_GATHER_CONFIG[(8, 128, 2)] = (32, 2)` in `vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.tuned` so `pallas::dsv41_packed_fp8_prepare_candidates` fits within TPU v6e's `256 KB` (`65,536`-word) SparseCore VMEM instead of overflowing (`int32[-256]`).
3. **TPU v6e Pallas Attention Path (`models/vllm/deepseek_v41/v2/attention.py`, `shifted_post_pre_rms.py`):** Gates `csa2_supported` on `pltpu.ChipVersion.TPU_7X` (since `csa2` uses `pltpu.ACC(mxu)` accumulators unavailable on v6e) so v6e routes through `sparse_fp8_attention` / `paged_attention`, and enables `ChipVersion.TPU_V6E` in `shifted_post_pre_rms.py`.
4. **XLA Large 2nd-Minor Layout Workaround (`serving/serve_node.sh`):** Adds `--xla_tpu_impure_enable_large_2nd_minor_layout=false --xla_tpu_impure_enable_large_2nd_minor_entry_layout=false --xla_tpu_impure_use_large_2nd_minor_layout_for_x16=false` to `LIBTPU_INIT_ARGS` to avoid an XLA `IsValidWindowForCopyFromLarge2ndMinorInput` `SIGILL` trap in `v41_encoder_tail`.
5. **Bottom-of-HBM Program Reservation & Bucket Cap (`runner/tpu_runner.py`, `worker/tpu_worker.py`, `scheduler.py`):** Skips the throwaway `kv=0` `profile_run` compilation (freeing a permanent `4.24 GiB` bottom-of-HBM program reservation), caps KV cache allocation at `5.05 GiB` (`933,080` tokens/rank), and caps per-rank scheduled tokens at `512` (`12,288` tokens cluster-wide across 24 DP ranks).
6. **Multi-Host Gloo Expert Parallelism (`platforms/tpu_platform.py`, `distributed/tpu_communicator.py`):** When `VLLM_TPU_GLOO_EP=1`, binds each worker to its local TPU v6e chip (`TPU_VISIBLE_CHIPS=0..3`) and splits Inductor graphs at `vllm::all_gather`, `vllm::reduce_scatter`, and `vllm::all_reduce` to execute multi-host collectives over 200 Gbps gVNIC TCP (`Gloo`).

Apply the patch across all hosts:
```bash
python3 infra/cluster_runner.py --sync --cmd 'python3 ~/deepseek-serving-tpu/infra/apply_multihost_gloo_ep_patch.py'
```

---

## 4. Launching Multi-Host `vllm serve` & Running Benchmarks

### 4.1 Start the Multi-Host vLLM Server
```bash
python3 serving/launch_multihost_vllm.py \
  --action restart \
  --execution-mode ced \
  --moe-backend gmm \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching 1
```

### 4.2 Run Functional Verification (`5/5` Test Suite)
```bash
ssh -i ~/.ssh/google_compute_engine <user>@<head-nat-ip> \
  'cd ~/deepseek-serving-tpu && python3 benchmarks/functional_test.py --out results/functional_verification.json'
```

### 4.3 Run Agentic Context-Length & Concurrency Benchmarks
```bash
ssh -i ~/.ssh/google_compute_engine <user>@<head-nat-ip> '
  source ~/vllm_tpu_env/bin/activate
  cd ~/deepseek-serving-tpu
  python3 benchmarks/agentic_benchmark.py --scenario agentic_4k_fp8 --input-len 4096 --output-len 32 --concurrencies 1,8,16,24,48 --num-prompts-multiplier 1 --out results/bench_agentic_4k_fp8.json
  python3 benchmarks/agentic_benchmark.py --scenario agentic_16k_fp8 --input-len 16384 --output-len 32 --concurrencies 1,8,16,24 --num-prompts-multiplier 1 --out results/bench_agentic_16k_fp8.json
  python3 benchmarks/agentic_benchmark.py --scenario agentic_16k_prefix_cached_fp8 --input-len 16384 --output-len 16 --shared-prefix-ratio 0.875 --concurrencies 1,8,24 --num-prompts-multiplier 1 --out results/bench_agentic_16k_prefix_cached_fp8.json
  python3 benchmarks/agentic_benchmark.py --scenario agentic_32k_fp8 --input-len 32768 --output-len 16 --concurrencies 1,8,24 --num-prompts-multiplier 1 --out results/bench_agentic_32k_fp8.json
  python3 benchmarks/agentic_benchmark.py --scenario agentic_64k_fp8 --input-len 64000 --output-len 16 --concurrencies 1,16 --num-prompts-multiplier 1 --out results/bench_agentic_64k_fp8.json
'
```

### 4.4 Run FP8 vs. BF16 TPU v6e Ablation & Generate Charts
```bash
# On a TPU v6e host (once vllm serve is stopped to release /dev/vfio/0):
ssh -i ~/.ssh/google_compute_engine <user>@<head-nat-ip> '
  source ~/vllm_tpu_env/bin/activate
  cd ~/deepseek-serving-tpu
  python3 benchmarks/fp8_vs_bf16_tpu_benchmark.py --out results/bench_ablation_fp8_vs_bf16.json
'

# Pull all results locally and generate CSV + PNG charts:
scp -i ~/.ssh/google_compute_engine "<user>@<head-nat-ip>:~/deepseek-serving-tpu/results/*.json" results/
python3 benchmarks/generate_charts.py --results-dir results
```

---

## 5. Cluster Teardown

Always tear down all TPU v6e VMs, instance templates, workload policies, and temporary regional GCS buckets when benchmarking finishes:

```bash
ZONE=asia-south1-c BASE_NAME=dsv41-as1c bash infra/teardown_tpu_v6e_mig.sh
```
