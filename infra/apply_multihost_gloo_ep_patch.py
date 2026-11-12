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

"""Idempotent patcher for vllm-torchtpu to enable Multi-Host Gloo Expert Parallelism (VLLM_TPU_GLOO_EP=1)
across standalone ct6e-standard-4t TPU v6e VMs connected over 200 Gbps gVNIC.

What this patch enables when `VLLM_TPU_GLOO_EP=1`:
1. Initializes each worker's local physical TPU v6e chip (`binding.native_local_rank` in 0..3)
   via a single-chip PjRtCApiClient before `vllm`'s `init_distributed_environment`, avoiding
   cross-VM optical ICI `tpunetd` slice-builder failures on standalone GCE TPU VMs.
2. Uses `gloo` for `vllm`'s distributed process groups across all multi-host DP/EP ranks.
3. Enables `TpuPlatform.use_custom_op_collectives() -> True` and splits the Inductor FX graph
   at `["vllm::all_gather", "vllm::reduce_scatter", "vllm::all_reduce"]` so each layer's
   TPU compute subgraphs compile once and hit TorchTPU's Tier-2/Tier-3 MLIR compilation cache,
   while `TpuDeviceCommunicator` executes host-staged `all_gather`, `reduce_scatter`, and
   `all_reduce` over `self.cpu_group` (Gloo).
"""

from pathlib import Path
import sys

REPO = Path.home() / "vllm-torchtpu" / "src" / "vllm_torchtpu"


def patch_file(path: Path, target: str, replacement: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"[SKIP] {path.name} already patched ({marker})")
        return
    if target not in text:
        raise RuntimeError(f"Target snippet not found in {path}:\n{target}")
    path.write_text(text.replace(target, replacement, 1))
    print(f"[OK]   Patched {path.name} ({marker})")


def main() -> None:
    if not REPO.exists():
        print(f"Error: {REPO} does not exist", file=sys.stderr)
        sys.exit(1)

    # 1. Patch platforms/tpu_platform.py
    tpu_platform = REPO / "platforms" / "tpu_platform.py"
    patch_file(
        tpu_platform,
        target="""TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}""",
        replacement="""TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    4: "2,2,1",
    8: "2,4,1",
    12: "2,6,1",
    16: "4,4,1",
    24: "4,6,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}""",
        marker='12: "2,6,1"',
    )

    patch_file(
        tpu_platform,
        target='        "TPU_MULTIHOST_BACKEND",\n',
        replacement='        "TPU_MULTIHOST_BACKEND",\n        "VLLM_TPU_GLOO_EP",\n',
        marker='"VLLM_TPU_GLOO_EP"',
    )

    patch_file(
        tpu_platform,
        target="""    @classmethod
    def get_worker_distributed_backend(cls, world_size: int) -> str:
        \"\"\"Pick torch.distributed backend used by worker bootstrap.

        TPU collectives are not needed when world_size==1, and single-rank
        `tpu_dist` bootstrap is unstable in CI. Use `gloo` in that case while
        keeping `tpu_dist` for multi-rank runs.
        \"\"\"
        if world_size == 1 and cls.dist_backend == "tpu_dist":
            return "gloo"
        return cls.dist_backend""",
        replacement="""    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1"

    @classmethod
    def get_worker_distributed_backend(cls, world_size: int) -> str:
        \"\"\"Pick torch.distributed backend used by worker bootstrap.

        TPU collectives are not needed when world_size==1, and single-rank
        `tpu_dist` bootstrap is unstable in CI. Use `gloo` in that case while
        keeping `tpu_dist` for multi-rank runs.
        \"\"\"
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1":
            return "gloo"
        if world_size == 1 and cls.dist_backend == "tpu_dist":
            return "gloo"
        return cls.dist_backend""",
        marker="def use_custom_op_collectives",
    )

    patch_file(
        tpu_platform,
        target="""        # No graph splitting — TPU handles the full graph including
        compilation_config.splitting_ops = []""",
        replacement="""        # No graph splitting — TPU handles the full graph including
        compilation_config.splitting_ops = []
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1":
            compilation_config.splitting_ops = [
                "vllm::all_gather",
                "vllm::reduce_scatter",
                "vllm::all_reduce",
            ]""",
        marker='"vllm::all_gather"',
    )

    # 2. Patch compilation/tpu_compiler.py
    tpu_compiler = REPO / "compilation" / "tpu_compiler.py"
    patch_file(
        tpu_compiler,
        target='    "TPU_MOE_EPLB_MOVER_STEP_BUDGET_MB",\n}',
        replacement='    "TPU_MOE_EPLB_MOVER_STEP_BUDGET_MB",\n    "VLLM_TPU_GLOO_EP",\n}',
        marker='"VLLM_TPU_GLOO_EP"',
    )

    # 3. Patch __init__.py (_patch_multiproc_worker_global_rank_env)
    init_py = REPO / "__init__.py"
    patch_file(
        init_py,
        target="""    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        binding = get_tpu_worker_binding(pc, rank, local_rank, env=_os.environ)
        _os.environ.update(binding.as_env())""",
        replacement="""    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        binding = get_tpu_worker_binding(pc, rank, local_rank, env=_os.environ)
        _os.environ.update(binding.as_env())
        if _os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1":
            _os.environ["TPU_VISIBLE_CHIPS"] = str(binding.native_local_rank)
            _os.environ["TPU_VISIBLE_DEVICES"] = str(binding.native_local_rank)
            _os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1" """,
        marker='if _os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1":',
    )

    # 4. Patch worker/tpu_worker.py
    tpu_worker = REPO / "worker" / "tpu_worker.py"
    patch_file(
        tpu_worker,
        target="""        self.profile_rank = binding.rank
        self.profile_world_size = binding.world_size
        if dp_size > 1:""",
        replacement="""        self.profile_rank = binding.rank
        self.profile_world_size = binding.world_size
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1":
            os.environ["TPU_VISIBLE_CHIPS"] = str(binding.native_local_rank)
            os.environ["TPU_VISIBLE_DEVICES"] = str(binding.native_local_rank)
            os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
            os.environ["TORCH_TPU_TOPOLOGY"] = "1,1,1"
            os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ.pop("WORLD_SIZE", None)
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            _ = torch.ones((1,), device="tpu").cpu()
            logger.info(
                "Initialized local PjRtCApiClient on chip=%d for VLLM_TPU_GLOO_EP rank=%d/%d",
                binding.native_local_rank,
                binding.rank,
                binding.world_size,
            )
        if dp_size > 1:""",
        marker="Initialized local PjRtCApiClient on chip=",
    )

    # 5. Patch distributed/tpu_communicator.py
    tpu_comm = REPO / "distributed" / "tpu_communicator.py"
    patch_file(
        tpu_comm,
        target="""    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._token_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states
        return group.reduce_scatter(hidden_states, dim=0)""",
        replacement="""    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        import os
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1" and input_.device.type == "tpu":
            cpu_t = input_.to("cpu").contiguous()
            torch.distributed.all_reduce(cpu_t, group=self.cpu_group)
            return cpu_t.to(input_.device)
        return super().all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        import os
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1" and input_.device.type == "tpu":
            if dim < 0:
                dim += input_.dim()
            cpu_in = input_.to("cpu").contiguous()
            input_size = cpu_in.size()
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
            cpu_out = torch.empty(output_size, dtype=cpu_in.dtype, device="cpu")
            torch.distributed.all_gather_into_tensor(
                cpu_out, cpu_in, group=self.cpu_group
            )
            cpu_out = cpu_out.reshape((self.world_size,) + input_size)
            cpu_out = cpu_out.movedim(0, dim)
            cpu_out = cpu_out.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            ).contiguous()
            return cpu_out.to(input_.device)
        return super().all_gather(input_, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        import os
        if os.environ.get("VLLM_TPU_GLOO_EP", "0") == "1" and input_.device.type == "tpu":
            if dim < 0:
                dim += input_.dim()
            cpu_in = input_.movedim(0, dim).to("cpu").contiguous()
            assert cpu_in.shape[0] % self.world_size == 0
            chunk_size = cpu_in.shape[0] // self.world_size
            output_shape = (chunk_size,) + cpu_in.shape[1:]
            cpu_out = torch.empty(output_shape, dtype=cpu_in.dtype, device="cpu")
            torch.distributed.reduce_scatter_tensor(
                cpu_out, cpu_in, group=self.cpu_group
            )
            cpu_out = cpu_out.movedim(0, dim).contiguous()
            return cpu_out.to(input_.device)
        return super().reduce_scatter(input_, dim=dim)

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._token_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states
        return group.reduce_scatter(hidden_states, dim=0)""",
        marker="torch.distributed.reduce_scatter_tensor(",
    )

    # 6. Patch kernels/deepseek_v41/mhc/shifted_post_pre_rms.py for TPU v6e (Trillium)
    shifted_mhc = REPO / "kernels" / "deepseek_v41" / "mhc" / "shifted_post_pre_rms.py"
    patch_file(
        shifted_mhc,
        target='    assert info.chip_version == pltpu.ChipVersion.TPU_7X, "Requires TPU7x"',
        replacement=(
            "    assert info.chip_version in ("
            "pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V6E"
            '), "Requires TPU7x or TPU v6e"'
        ),
        marker="pltpu.ChipVersion.TPU_V6E",
    )

    # 7. Patch layers/core/fused_moe_gmm.py to use native packed int4 (W4A8 INT8xINT4 MXU)
    #    on TPU v6e (generation < 7, which does not have hardware float4_e2m1fn unpack)
    fused_moe_gmm = REPO / "layers" / "core" / "fused_moe_gmm.py"
    patch_file(
        fused_moe_gmm,
        target="""    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(
        *w.shape[:-2], num_blocks, block, w.shape[-1]
    )
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)""",
        replacement="""    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(
        *w.shape[:-2], num_blocks, block, w.shape[-1]
    )
    if pack and pltpu.get_tpu_info().generation < 7:
        int4_max = 7.0
        abs_max = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True)
        scale = (abs_max / int4_max).astype(jnp.float32)
        scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
        quantized = jnp.round(
            jnp.clip(blocked * scale_inv, -int4_max, int4_max)
        ).astype(jnp.int4)
        quantized = quantized.reshape(*w.shape[:-2], size_k, w.shape[-1])
        pairs = quantized.reshape(*w.shape[:-2], size_k // 2, 2, w.shape[-1])
        pairs = jnp.swapaxes(pairs, -1, -2)
        packed = jax.lax.bitcast_convert_type(pairs, jnp.uint8)
        return packed, scale
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)""",
        marker="if pack and pltpu.get_tpu_info().generation < 7:",
    )

    # 8. Patch layers/adapter/quantization/mxfp4.py so VllmDeepseekV4Mxfp4MoEMethod
    #    uses rhs_quant_dtype=jnp.int4 on TPU v6e (generation < 7)
    mxfp4 = REPO / "layers" / "adapter" / "quantization" / "mxfp4.py"
    patch_file(
        mxfp4,
        target="""        if op is None:
            prebuild_fused_moe_kernel(
                topk=layer.moe_config.experts_per_token,
                activation=activation_str,
                use_ep=layer.moe_config.moe_parallel_config.use_ep,
                rhs_quant_dtype=jnp.float4_e2m1fn,
            )""",
        replacement="""        if op is None:
            from jax.experimental.pallas import tpu as pltpu
            self.rhs_quant_dtype = (
                jnp.int4
                if pltpu.get_tpu_info().generation < 7
                else jnp.float4_e2m1fn
            )
            prebuild_fused_moe_kernel(
                topk=layer.moe_config.experts_per_token,
                activation=activation_str,
                use_ep=layer.moe_config.moe_parallel_config.use_ep,
                rhs_quant_dtype=self.rhs_quant_dtype,
            )""",
        marker="pltpu.get_tpu_info().generation < 7",
    )

    # 9. Patch vllm/compilation/codegen.py so operator.getitem with non-int (slice) indices
    #    and _operator.* functions inside inlined split_gm submodules compile cleanly
    vllm_codegen = (
        Path.home()
        / "vllm_tpu_env"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm"
        / "compilation"
        / "codegen.py"
    )
    if vllm_codegen.exists():
        cg_text = vllm_codegen.read_text()
        cg_text = cg_text.replace(
            "            if node.target is operator.getitem and isinstance(node.args[1], int):\n                source = ref(node.args[0])\n                index = node.args[1]\n                assert isinstance(index, int)",
            "            if node.target is operator.getitem:\n                source = ref(node.args[0])\n                index = ref(node.args[1])",
        )
        cg_text = cg_text.replace(
            "            if node.target is operator.getitem:\n                source = ref(node.args[0])\n                index = node.args[1]\n                assert isinstance(index, int)",
            "            if node.target is operator.getitem:\n                source = ref(node.args[0])\n                index = ref(node.args[1])",
        )
        if "import operator as _operator" not in cg_text:
            cg_text = cg_text.replace(
                'return "import torch\\nimport operator\\n" + code, submod_names, consts',
                'return "import torch\\nimport operator\\nimport operator as _operator\\n" + code, submod_names, consts',
            )
        vllm_codegen.write_text(cg_text)
        print("[OK]   Patched codegen.py (ref(node.args[1]) + import operator as _operator)")

    # 10. Patch models/vllm/deepseek_v41/v2/attention.py so csa2_supported is only enabled on
    #     TPU 7x (since csa2 uses pltpu.ACC(mxu) which is only on TPU 7x; TPU v6e uses the
    #     native sparse_fp8_attention / paged_attention path).
    dsv41_attn = REPO / "models" / "vllm" / "deepseek_v41" / "v2" / "attention.py"
    patch_file(
        dsv41_attn,
        target="""        csa2_supported = (
            get_tensor_model_parallel_world_size() == 1
            and self._fp8_kv
            and self.head_dim == 512
            and self.window_size == 128
            and self.n_local_heads == 64
        )""",
        replacement="""        from jax.experimental.pallas import tpu as pltpu
        from vllm_torchtpu.tpu_info import get_chip_version
        csa2_supported = (
            get_tensor_model_parallel_world_size() == 1
            and self._fp8_kv
            and self.head_dim == 512
            and self.window_size == 128
            and self.n_local_heads == 64
            and get_chip_version() == pltpu.ChipVersion.TPU_7X
        )""",
        marker="and get_chip_version() == pltpu.ChipVersion.TPU_7X",
    )

    # 11. Patch compilation/tpu_compiler.py so large_2nd_minor XLA flags in LIBTPU_INIT_ARGS
    #     preserve the existing compile cache hash for already-compiled stages.
    tpu_compiler = REPO / "compilation" / "tpu_compiler.py"
    patch_file(
        tpu_compiler,
        target='        "native_env": {name: os.getenv(name) for name in _NATIVE_TPU_COMPILE_ENV_VARS},',
        replacement="""        "native_env": {
            name: (
                " ".join(
                    tok
                    for tok in (os.getenv(name) or "").split()
                    if "large_2nd_minor" not in tok
                )
                if name == "LIBTPU_INIT_ARGS" and os.getenv(name) is not None
                else os.getenv(name)
            )
            for name in _NATIVE_TPU_COMPILE_ENV_VARS
        },""",
        marker='if "large_2nd_minor" not in tok',
    )

    # 12. Patch worker/tpu_worker.py to set TPU v6e SparseCore 256KB VMEM-compatible
    #     gather tile/buffer config in fp8.tuned at worker init without altering kernel file hashes.
    patch_file(
        tpu_worker,
        target="""            logger.info(
                "Initialized local PjRtCApiClient on chip=%d for VLLM_TPU_GLOO_EP rank=%d/%d",
                binding.native_local_rank,
                binding.rank,
                binding.world_size,
            )""",
        replacement="""            from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import (
                tuned as _fp8_tuned,
            )
            _fp8_tuned.PAGED_MQA_GATHER_CONFIG[(8, 1)] = (32, 2)
            _fp8_tuned.PAGED_MQA_GATHER_CONFIG[(8, 2)] = (32, 2)
            _fp8_tuned.PAGED_MQA_FRAGMENT_GATHER_CONFIG[(8, 128, 2)] = (32, 2)
            logger.info(
                "Initialized local PjRtCApiClient on chip=%d for VLLM_TPU_GLOO_EP rank=%d/%d",
                binding.native_local_rank,
                binding.rank,
                binding.world_size,
            )""",
        marker="_fp8_tuned.PAGED_MQA_GATHER_CONFIG[(8, 1)] = (32, 2)",
    )

    # 13. Patch runner/tpu_runner.py to skip the throwaway kv=0 profile_run _dummy_run on
    #     32GB TPU v6e when VLLM_TPU_SKIP_PROFILE_DUMMY=1 so we don't leak a permanent 4.24 GiB
    #     bottom-of-HBM program-region reservation before _precompile_backbone loads the real
    #     serving executables.
    tpu_runner = REPO / "runner" / "tpu_runner.py"
    patch_file(
        tpu_runner,
        target="""            with self._profile_isolated_cache():
                cc.compile_sizes = [num_tokens]
                self._dummy_run(
                    num_tokens, self.num_reqs_max_model_len, self.max_num_blocks_per_req
                )
            reset_compile_wrapper(self.model)
            # reset_compile_wrapper wipes cache_dir; restore so
            # capture_model can read/write the persistent cache.
            cc.cache_dir, cc.local_cache_dir = saved_cache
            cc.compile_sizes = saved_sizes
            torch._dynamo.reset()
            synchronize_device()""",
        replacement="""            if os.environ.get("VLLM_TPU_SKIP_PROFILE_DUMMY", "1") != "1":
                with self._profile_isolated_cache():
                    cc.compile_sizes = [num_tokens]
                    self._dummy_run(
                        num_tokens, self.num_reqs_max_model_len, self.max_num_blocks_per_req
                    )
                reset_compile_wrapper(self.model)
                # reset_compile_wrapper wipes cache_dir; restore so
                # capture_model can read/write the persistent cache.
                cc.cache_dir, cc.local_cache_dir = saved_cache
                cc.compile_sizes = saved_sizes
                torch._dynamo.reset()
                synchronize_device()""",
        marker='VLLM_TPU_SKIP_PROFILE_DUMMY',
    )

    # 14. Patch worker/tpu_worker.py so when VLLM_TPU_SKIP_PROFILE_DUMMY=1, available HBM for
    #     KV cache is capped at VLLM_TPU_KV_AVAIL_GB (default 5.05 GiB), preserving the exact
    #     5,368,709,120-byte (933,080-token) KV cache shape compiled in ac42ffdf87 while keeping
    #     the saved 4.24 GiB bottom-of-HBM program headroom for _precompile_backbone.
    patch_file(
        tpu_worker,
        target="        available = budget.available - kv_connector_hbm_reserve\n",
        replacement=(
            "        available = budget.available - kv_connector_hbm_reserve\n"
            '        if os.environ.get("VLLM_TPU_SKIP_PROFILE_DUMMY", "1") == "1":\n'
            "            available = min(\n"
            "                available,\n"
            '                int(float(os.environ.get("VLLM_TPU_KV_AVAIL_GB", "5.05")) * utils.GBYTES),\n'
            "            )\n"
        ),
        marker="VLLM_TPU_KV_AVAIL_GB",
    )

    # 15. Patch runner/tpu_runner.py (_precompile_backbone) so on 32GB TPU v6e with 24-way
    #     AllGather MoE, warmup caps active token buckets at VLLM_TPU_MAX_WARMUP_TOKENS (512
    #     tokens/rank = 12,288 tokens cluster-wide) AFTER PiecewiseBackend has loaded the
    #     ac42ffdf87 compile cache during the 64-token trace.
    patch_file(
        tpu_runner,
        target="""        with shape_variants.warmup() as buckets:
            for num_tokens in self.num_tokens_paddings:
                logger.info("  -- num_tokens: %d", num_tokens)""",
        replacement="""        _max_warmup_tokens = int(os.environ.get("VLLM_TPU_MAX_WARMUP_TOKENS", "512"))
        with shape_variants.warmup() as buckets:
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > _max_warmup_tokens:
                    continue
                logger.info("  -- num_tokens: %d", num_tokens)""",
        marker="VLLM_TPU_MAX_WARMUP_TOKENS",
    )
    patch_file(
        tpu_runner,
        target="""        if self._pp_wave is not None:
            self._warmup_handoff_forwards()""",
        replacement="""        self.num_tokens_paddings = [
            n for n in self.num_tokens_paddings if n <= _max_warmup_tokens
        ]
        self.max_num_tokens = self.num_tokens_paddings[-1]
        if self._pp_wave is not None:
            self._warmup_handoff_forwards()""",
        marker="n for n in self.num_tokens_paddings if n <= _max_warmup_tokens",
    )

    # 16. Patch vllm/v1/core/sched/scheduler.py so the V1 chunked-prefill scheduler caps
    #     per-rank scheduled tokens at VLLM_TPU_MAX_WARMUP_TOKENS (512 tokens/rank = 12,288
    #     tokens cluster-wide) without changing VllmConfig's compile_sizes hash.
    vllm_sched = (
        Path.home()
        / "vllm_tpu_env"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm"
        / "v1"
        / "core"
        / "sched"
        / "scheduler.py"
    )
    if vllm_sched.exists():
        patch_file(
            vllm_sched,
            target="""        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )""",
            replacement="""        import os as _os
        _max_tpu_tok = int(_os.environ.get("VLLM_TPU_MAX_WARMUP_TOKENS", "512"))
        self.scheduler_config.max_num_batched_tokens = min(
            self.scheduler_config.max_num_batched_tokens, _max_tpu_tok
        )
        self.max_num_scheduled_tokens = min(
            (
                self.scheduler_config.max_num_scheduled_tokens
                if self.scheduler_config.max_num_scheduled_tokens is not None
                else self.scheduler_config.max_num_batched_tokens
            ),
            _max_tpu_tok,
        )""",
            marker="_max_tpu_tok",
        )


if __name__ == "__main__":
    main()
