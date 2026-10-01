#!/usr/bin/env python3
"""Time a whole attention block on one prefill cell, hidden states in and out.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/attention_block.py \\
        --impl flywheel --heads 32 --heads-k 32 --head-dim 256 --seq 16384 \\
        --output results/benchmark/softmax_attention/benchmark.jsonl \\
        --trace results/benchmark/softmax_attention/traces

Every impl starts from x (batch, seq, d_model) and ends at (batch, seq,
d_model), with d_model = heads * head_dim, so the qkv projection, whatever
layout conversion the kernel's input needs, the kernel and the output
projection are all inside the clock:

  flywheel  einsum -> (batch, heads, seq, head_dim) -> flash_attn_func
  splash    einsum -> (batch, heads, seq, head_dim), heads folded into batch
            -> splash attention
  rpa       einsum -> (batch * seq, heads, head_dim) -> ragged_paged_attention,
            which also writes the paged KV cache
  batched_rpa  the same inputs and paged cache -> tpu-inference's experimental
            batched RPA kernel, which also builds its tile schedule in a
            second Pallas kernel

Each projection is one einsum against a (d_model, heads, head_dim) weight that
emits the shape its kernel takes, as vLLM's attention layer projects, so any
relayout left in the block is one XLA could not fold into the projection.

With --trace the cell also runs under jax.profiler, and segments.py splits the
device time of that same executable into the qkv projection, layout, the
attention kernel and the output projection. The kernel segment is the
kernel-level number (kernel_tflops); the wall clock is the block-level one.

Block sizes come from each impl's table for the chip the cell runs on:
splash, RPA and batched RPA from splash_tuned_<chip>.json, rpa_tuned_<chip>.json
and batched_rpa_tuned_<chip>.json next to this file (v6e or v7x; tune_blocks.py
searches them), flywheel from flywheel_tpu's own lookup. A cell its table does
not hold runs the impl's own default: RPA's formula, Tokamax's heuristic,
batched RPA's VMEM-budget calculation. A cell already recorded as ok in
--output is skipped unless --trace is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np

from benchmarks.common.batched_rpa import FIELDS as BATCHED_RPA_FIELDS
from benchmarks.common.batched_rpa import load_batched_rpa
from benchmarks.common.records import recorded, write_record
from benchmarks.common.rpa import DEFAULT_RPA_SOURCE, load_rpa_v3, vllm_page_size
from benchmarks.common.rpa import FIELDS as RPA_FIELDS
from benchmarks.common.splash import (
    DEFAULT_SPLASH_SOURCE,
    HEURISTIC,
    load_splash,
)
from benchmarks.common.splash import FIELDS as SPLASH_FIELDS
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention import segments as segments_lib

RPA_TABLE = pathlib.Path(__file__).with_name("rpa_tuned_v6e.json")
SPLASH_TABLE = pathlib.Path(__file__).with_name("splash_tuned_v6e.json")
# device_kind -> the suffix of the tables searched on that chip; any other
# chip reads the v6e tables.
TABLE_TAGS = {"TPU7x": "v7x"}
IMPLS = ("flywheel", "splash", "rpa", "batched_rpa")
TRACE_CALLS = 3


@dataclasses.dataclass(frozen=True)
class Cell:
    impl: str
    heads: int
    heads_k: int
    head_dim: int
    mask: str
    batch: int
    seq: int

    @property
    def causal(self):
        return self.mask == "causal"

    @property
    def d_model(self):
        return self.heads * self.head_dim

    @property
    def key(self):
        return (f"{self.impl}-h{self.heads}-k{self.heads_k}-d{self.head_dim}"
                f"-{self.mask}-b{self.batch}-s{self.seq}")


def tuned_table(impl, device_kind):
    """The path of impl's tuned table for this chip, which may not exist."""
    tag = TABLE_TAGS.get(device_kind, "v6e")
    return pathlib.Path(__file__).with_name(f"{impl}_tuned_{tag}.json")


def table_entry(path, cell, per_mask):
    """The tuned table's entry for this cell as {field: value}, or None.

    The RPA table has no mask level, the splash table does.
    """
    if not path.exists():
        return None
    table = json.loads(path.read_text())
    head = (f"q_head-{cell.heads}_kv_head-{cell.heads_k}"
            f"_head-{cell.head_dim}")
    entry = table["configs"].get(head, {})
    if per_mask:
        entry = entry.get(cell.mask, {})
    values = entry.get(str(cell.seq))
    if values is None:
        return None
    order = table.get("config_order") or table["block_order"]
    return dict(zip(order, values))


def table_page_size(path, cell):
    """The KV page size the table searched this sequence length on, or None
    for vLLM's default."""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text()).get("page_size", {}).get(str(cell.seq))


def make_weights(cell):
    """x and the four projection weights, bf16, drawn on the host.

    Weights are scaled by 1/sqrt(d_model) so q, k and v come out near unit
    variance.
    """
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    seed = 0
    for part in (cell.heads, cell.heads_k, cell.head_dim, cell.batch,
                 cell.seq):
        seed = (seed * 1000003 + part) % (2**31)
    generator = np.random.default_rng(seed)

    def draw(*shape, scale=1.0):
        block = generator.standard_normal(shape, dtype=np.float32) * scale
        return jnp.asarray(block.astype(ml_dtypes.bfloat16))

    fan = 1.0 / math.sqrt(cell.d_model)
    return jax.block_until_ready({
        "x": draw(cell.batch, cell.seq, cell.d_model),
        "wq": draw(cell.d_model, cell.heads, cell.head_dim, scale=fan),
        "wk": draw(cell.d_model, cell.heads_k, cell.head_dim, scale=fan),
        "wv": draw(cell.d_model, cell.heads_k, cell.head_dim, scale=fan),
        "wo": draw(cell.heads, cell.head_dim, cell.d_model, scale=fan),
    })


def build_flywheel(cell, w, args):
    """(step, fresh_state, info) for flash_attn_func on head-major q, k, v."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu import flash_attn_func

    def block(x, wq, wk, wv, wo):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->bnth", x, wq)
            k = jnp.einsum("btd,dnh->bnth", x, wk)
            v = jnp.einsum("btd,dnh->bnth", x, wv)
        with jax.named_scope("attn"):
            out = flash_attn_func(q, k, v, causal=cell.causal)
        with jax.named_scope("out_proj"):
            return jnp.einsum("bnth,nhd->btd", out, wo)

    call = jax.jit(block)
    operands = (w["x"], w["wq"], w["wk"], w["wv"], w["wo"])
    return (lambda state: (call(*operands), state)), (lambda: None), {}


def build_splash(cell, w, args):
    """(step, fresh_state, info) for splash attention on (batch * heads, ...)."""
    import jax
    import jax.numpy as jnp

    kernel, mask_lib = load_splash(args.splash_source)
    config = table_entry(args.splash_table, cell, per_mask=True)
    source = "table"
    if config is None:
        config, source = dict(HEURISTIC), "heuristic"

    layouts = {"h": kernel.QKVLayout.HEAD_DIM_MINOR,
               "s": kernel.QKVLayout.SEQ_MINOR}
    diag_grid = int(config["diag_grid"])
    splash_config = kernel.SplashConfig(
        block_q=int(config["block_q"]),
        block_kv=int(config["block_kv"]),
        block_kv_compute=int(config["block_kv_compute"]),
        num_stacked_q_heads=int(config["num_stacked_q_heads"]),
        q_layout=layouts[config["layout"][0]],
        k_layout=layouts[config["layout"][1]],
        v_layout=layouts[config["layout"][2]],
        use_experimental_scheduler=bool(config["scheduler"]),
        qk_diag_skip=diag_grid > 0,
        sv_diag_skip=diag_grid > 0,
        qk_diag_grid=diag_grid or 2)
    shape = (cell.seq, cell.seq)
    mask = (mask_lib.CausalMask(shape) if cell.causal
            else mask_lib.FullMask(shape))
    attention = kernel.make_splash_mha_single_device(mask, config=splash_config)

    # Splash takes no softmax scale, so it goes into wq, outside the clock.
    wq = (w["wq"].astype(jnp.float32)
          / math.sqrt(cell.head_dim)).astype(jnp.bfloat16)

    def block(x, wq, wk, wv, wo):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->bnth", x, wq)
            k = jnp.einsum("btd,dnh->bnth", x, wk)
            v = jnp.einsum("btd,dnh->bnth", x, wv)
        with jax.named_scope("attn"):
            out = attention(*(operand.reshape(-1, *operand.shape[2:])
                              for operand in (q, k, v))).reshape(q.shape)
        with jax.named_scope("out_proj"):
            return jnp.einsum("bnth,nhd->btd", out, wo)

    call = jax.jit(block)
    operands = (w["x"], wq, w["wk"], w["wv"], w["wo"])
    info = {"config_source": source,
            "config": [config[name] for name in SPLASH_FIELDS]}
    return (lambda state: (call(*operands), state)), (lambda: None), info


def build_rpa(cell, w, args):
    """(step, fresh_state, info) for ragged_paged_attention on a paged cache."""
    import jax
    import jax.numpy as jnp

    rpa = load_rpa_v3(args.rpa_source)
    total = cell.batch * cell.seq
    page_size = args.page_size or vllm_page_size(cell.seq, cell.batch)
    pages_per_seq = -(-cell.seq // page_size)
    num_pages = cell.batch * pages_per_seq
    cache_shape = rpa.get_kv_cache_shape(num_pages, page_size, cell.heads_k,
                                         cell.head_dim, jnp.bfloat16)
    metadata = (
        jnp.full((cell.batch,), cell.seq, jnp.int32),
        jnp.arange(num_pages, dtype=jnp.int32),
        jnp.arange(cell.batch + 1, dtype=jnp.int32) * cell.seq,
        jnp.asarray(np.array([0, 0, cell.batch], np.int32)),
    )

    blocks = table_entry(args.rpa_table, cell, per_mask=False)
    source = "table"
    if blocks is None:
        formula = rpa.get_default_block_sizes(
            jnp.bfloat16, jnp.bfloat16, cell.heads, cell.heads_k,
            cell.head_dim, page_size, total, cell.batch, pages_per_seq,
            case=rpa.RpaCase.MIXED)
        blocks = {name: int(formula[name]) for name in RPA_FIELDS}
        source = "formula"
    block_sizes = tuple(int(blocks[name]) for name in RPA_FIELDS)

    kernel = getattr(rpa.ragged_paged_attention, "__wrapped__",
                     rpa.ragged_paged_attention)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def block(x, wq, wk, wv, wo, cache):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->btnh", x, wq).reshape(
                total, cell.heads, cell.head_dim)
            k = jnp.einsum("btd,dnh->btnh", x, wk).reshape(
                total, cell.heads_k, cell.head_dim)
            v = jnp.einsum("btd,dnh->btnh", x, wv).reshape(
                total, cell.heads_k, cell.head_dim)
        with jax.named_scope("attn"):
            out, cache = kernel(q, k, v, cache, *metadata,
                                use_causal_mask=cell.causal, sm_scale=scale,
                                m_block_sizes=block_sizes)
        with jax.named_scope("out_proj"):
            out = jnp.einsum(
                "btnh,nhd->btd",
                out.reshape(cell.batch, cell.seq, cell.heads, cell.head_dim),
                wo)
        return out, cache

    # The call donates the cache, so every timing run starts from a fresh one.
    call = jax.jit(block, donate_argnums=(5,))
    operands = (w["x"], w["wq"], w["wk"], w["wv"], w["wo"])
    info = {"block_source": source, "blocks": list(block_sizes),
            "page_size": page_size}
    return ((lambda state: call(*operands, state)),
            (lambda: jnp.zeros(cache_shape, jnp.bfloat16)), info)


def build_batched_rpa(cell, w, args):
    """(step, fresh_state, info) for the experimental batched RPA kernel."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    wrapper, configs, tuned_params = load_batched_rpa(args.rpa_source)
    if not cell.causal:
        raise ValueError("batched RPA only supports causal attention.")
    total = cell.batch * cell.seq
    # Note: on v7x the kernel halts the core on the 16-token pages vLLM uses
    # past 8192 tokens, so its table records the page size of those cells.
    page_size = (args.page_size
                 or table_page_size(args.batched_rpa_table, cell)
                 or vllm_page_size(cell.seq, cell.batch))
    pages_per_seq = -(-cell.seq // page_size)
    num_pages = cell.batch * pages_per_seq
    # Note: the layout tpu-inference runs unless USE_BATCHED_RPA_SEQ_ON_LANE is
    # set; SEQ_ALONG_LANE only supports a page size of 128.
    layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    cache_shape = wrapper.get_kv_cache_shape(
        num_pages, page_size, cell.heads_k, cell.head_dim, jnp.bfloat16,
        kv_layout=layout)
    metadata = (
        jnp.full((cell.batch,), cell.seq, jnp.int32),
        jnp.arange(num_pages, dtype=jnp.int32),
        jnp.arange(cell.batch + 1, dtype=jnp.int32) * cell.seq,
        jnp.asarray(np.array([0, 0, cell.batch], np.int32)),
    )
    scale = 1.0 / math.sqrt(cell.head_dim)

    # Note: without a table entry these are the block sizes the wrapper would
    # pick on its own, resolved here so the record holds what ran.
    source = "calculated"
    blocks = tuned_params.get_tuned_params(
        configs.ModelConfigs(
            num_q_heads=cell.heads, num_kv_heads=cell.heads_k,
            head_dim=cell.head_dim, sm_scale=scale,
            mask_value=float(jnp.finfo(jnp.bfloat16).min)),
        configs.ServingConfigs(
            num_seqs=cell.batch, num_page_indices=num_pages,
            total_q_tokens=total, dtype_q=jnp.bfloat16, dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16, page_size=page_size, kv_layout=layout),
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        case="prefill")
    tuned = table_entry(args.batched_rpa_table, cell, per_mask=False)
    if tuned is not None:
        blocks = configs.BlockSizes(
            **{name: int(tuned[name]) for name in BATCHED_RPA_FIELDS})
        source = "table"

    kernel = getattr(wrapper.ragged_paged_attention, "__wrapped__",
                     wrapper.ragged_paged_attention)

    def block(x, wq, wk, wv, wo, cache):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->btnh", x, wq).reshape(
                total, cell.heads, cell.head_dim)
            k = jnp.einsum("btd,dnh->btnh", x, wk).reshape(
                total, cell.heads_k, cell.head_dim)
            v = jnp.einsum("btd,dnh->btnh", x, wv).reshape(
                total, cell.heads_k, cell.head_dim)
        with jax.named_scope("attn"):
            out, cache = kernel(q, k, v, cache, *metadata, sm_scale=scale,
                                prefill_block_sizes=blocks, kv_layout=layout)
        with jax.named_scope("out_proj"):
            out = jnp.einsum(
                "btnh,nhd->btd",
                out.reshape(cell.batch, cell.seq, cell.heads, cell.head_dim),
                wo)
        return out, cache

    # The call donates the cache, so every timing run starts from a fresh one.
    call = jax.jit(block, donate_argnums=(5,))
    operands = (w["x"], w["wq"], w["wk"], w["wv"], w["wo"])
    info = {"block_source": source,
            "blocks": [getattr(blocks, name) for name in BATCHED_RPA_FIELDS],
            "page_size": page_size, "kv_layout": str(layout)}
    return ((lambda state: call(*operands, state)),
            (lambda: jnp.zeros(cache_shape, jnp.bfloat16)), info)


BUILDERS = {"flywheel": build_flywheel, "splash": build_splash,
            "rpa": build_rpa, "batched_rpa": build_batched_rpa}


def trace_cell(cell, step, state, directory):
    """Run a few calls under jax.profiler and split the trace, or None."""
    import jax

    target = directory / cell.key
    target.mkdir(parents=True, exist_ok=True)
    out, state = step(state)
    jax.block_until_ready(out)
    with jax.profiler.trace(str(target)):
        for _ in range(TRACE_CALLS):
            out, state = step(state)
        jax.block_until_ready(out)
    trace = segments_lib.load(target)
    return segments_lib.segments(trace) if trace is not None else None


def run(cell, args):
    import jax

    pair_count = cell.batch * cell.seq * cell.seq
    if cell.causal:
        pair_count //= 2
    attn_flops = 4 * pair_count * cell.heads * cell.head_dim
    proj_flops = (2 * cell.batch * cell.seq * cell.d_model
                  * (cell.heads + 2 * cell.heads_k) * cell.head_dim
                  + 2 * cell.batch * cell.seq * cell.heads * cell.head_dim
                  * cell.d_model)
    record = {"key": cell.key, "cell": dataclasses.asdict(cell),
              "d_model": cell.d_model, "attn_flops": attn_flops,
              "proj_flops": proj_flops}
    try:
        weights = make_weights(cell)
        step, fresh, info = BUILDERS[cell.impl](cell, weights, args)
        record.update(info)
        timing, _ = time_call(step, fresh())
    except Exception as error:
        record.update(status="error",
                      error=f"{type(error).__name__}: {str(error)[:300]}")
        return record

    ms = timing["median_ms"]
    record.update(status="ok", **timing,
                  block_tflops=(attn_flops + proj_flops) / (ms * 1e9))
    if args.trace:
        split = trace_cell(cell, step, fresh(), args.trace)
        if split is None:
            record["trace_error"] = "no device events in the trace"
        else:
            kernel_ms = split["ms"]["attn_kernel"]
            record.update(
                trace_module_ms=split["module_ms"],
                segments_ms=split["ms"],
                kernel_ms=kernel_ms,
                kernel_tflops=(attn_flops / (kernel_ms * 1e9)
                               if kernel_ms else None))
    return record


def main(argv=None):
    args = parse_args(argv)
    cell = Cell(args.impl, args.heads, args.heads_k, args.head_dim, args.mask,
                args.batch, args.seq)
    if not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded, skipped", flush=True)
        return 0

    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"requires TPU; got {jax.default_backend()!r}.")
    device_kind = jax.devices()[0].device_kind
    for impl in ("rpa", "splash", "batched_rpa"):
        if getattr(args, f"{impl}_table") is None:
            setattr(args, f"{impl}_table", tuned_table(impl, device_kind))

    record = run(cell, args)
    write_record(args.output, record)
    if record["status"] != "ok":
        print(f"{cell.key}: {record['status']} {record.get('error')}",
              flush=True)
        return 1
    line = f"{cell.key}: block {record['median_ms']:.3f} ms"
    if "segments_ms" in record:
        parts = "  ".join(f"{name} {value:.3f}"
                          for name, value in record["segments_ms"].items())
        line += (f" | {parts} | kernel {record['kernel_tflops']:.1f}"
                 " TFLOP/s")
    print(line, flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--mask", choices=("causal", "full"), default="causal")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16384)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--trace", type=pathlib.Path,
                        help="directory for the profiler traces, one "
                             "subdirectory per cell")
    parser.add_argument("--rpa-table", type=pathlib.Path,
                        help="default: rpa_tuned_<chip>.json next to this "
                             "file; a missing file selects RPA's formula")
    parser.add_argument("--splash-table", type=pathlib.Path,
                        help="default: splash_tuned_<chip>.json")
    parser.add_argument("--batched-rpa-table", type=pathlib.Path,
                        help="default: batched_rpa_tuned_<chip>.json")
    parser.add_argument("--page-size", type=int,
                        help="KV page size for rpa and batched_rpa instead of "
                             "vLLM's default for the sequence length")
    parser.add_argument("--rpa-source", type=pathlib.Path,
                        default=DEFAULT_RPA_SOURCE)
    parser.add_argument("--splash-source", type=pathlib.Path,
                        default=DEFAULT_SPLASH_SOURCE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
