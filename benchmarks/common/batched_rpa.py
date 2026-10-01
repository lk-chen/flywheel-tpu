"""SHA-pinned loader for tpu-inference's experimental batched RPA kernel.

--rpa-source defaults to the third_party/tpu-inference submodule
(git submodule update --init third_party/tpu-inference). Only
tpu_inference/kernels/experimental/batched_rpa/*.py and tpu_inference/envs.py
are loaded, and each is verified against its SHA-256 at RPA_V3_GIT_SHA, the
submodule pin the RPA v3 baseline uses.

The kernel has no tuned table: tuned_params.calculate_block_sizes derives its
block sizes from the chip's VMEM capacity, as tpu-inference does under
USE_BATCHED_RPA_KERNEL=1.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
import types
from pathlib import Path

from benchmarks.common.rpa import RPA_V3_GIT_SHA

BATCHED_RPA_PACKAGE = "tpu_inference.kernels.experimental.batched_rpa"
BATCHED_RPA_SOURCE_SHA256 = {
    "__init__.py":
        "e9b8af1551a068f26499e15c674fe09ca39fa39e96c7c93740c2f2afcf1a7588",
    "bref_override.py":
        "f64ecdefc69b7ed486e0921135241501275ce4234ee8bac342173b1e669823ac",
    "configs.py":
        "1941bee1fb5047fc92e932c781f10d0a95aa0dd087cc8e28d3639bbb4fb8fdf4",
    "flash_attention.py":
        "a9bb67f576bff5d59538b16509f7b52064844be53f1bf2d14358f0d1900ad658",
    "kernel.py":
        "d082659d779655b77e2cac7b2d427954217f3740d7de72ffc6c1527555fa8345",
    "schedule.py":
        "8ef5150972123b5a2dbb9ee2533fe0c0962448b3565157a649408034e41b9c0e",
    "stitch_utils.py":
        "fa3c17237707aa14be6a755dc31ac7e76252fe544730598bd27853a4fc6139ce",
    "tuned_params.py":
        "2ef84414885a431684b88661a0767396ea30625cb8ec6e1ff49ec12256df6f64",
    "utils.py":
        "0ffb823cdf662d2907a37866fe33df74c8504d9227d588b750d221cbab717dda",
    "wrapper.py":
        "5d6d6532c3519280e3c85fdbc988955c7aa6d86732645affa4f53287c9d73d67",
}
ENVS_SHA256 = "0f99a5055d8e3885cce4c3bf471e9b84683bb55db99d52622fd8a041be9d21f7"
FIELDS = ("bq_sz", "bq_c_sz", "bkv_sz", "batch_size", "n_buffer")


def _check(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"{path} does not match tpu-inference {RPA_V3_GIT_SHA}; got "
            f"sha256 {digest}.")


def validate_batched_rpa_source(source_root: Path) -> Path:
    """The pinned batched_rpa package directory inside a tpu-inference
    checkout."""
    source_root = Path(source_root).expanduser().resolve()
    module_root = source_root.joinpath(*BATCHED_RPA_PACKAGE.split("."))
    envs = source_root / "tpu_inference" / "envs.py"
    missing = [name for name in BATCHED_RPA_SOURCE_SHA256
               if not (module_root / name).is_file()]
    if missing or not envs.is_file():
        raise FileNotFoundError(
            f"{module_root} is missing {missing or [envs.name]}; --rpa-source "
            f"must be a tpu-inference checkout at {RPA_V3_GIT_SHA} (for the "
            f"default, run git submodule update --init "
            f"third_party/tpu-inference).")
    for name, expected in BATCHED_RPA_SOURCE_SHA256.items():
        _check(module_root / name, expected)
    _check(envs, ENVS_SHA256)
    return module_root


def load_batched_rpa(source_root: Path):
    """The pinned (wrapper, configs, tuned_params) modules, imported under stub
    parent packages so no tpu-inference install (or its vLLM dependency) is
    needed."""
    module_root = validate_batched_rpa_source(source_root)
    if "tpu_inference" in sys.modules:
        raise RuntimeError(
            "tpu_inference was imported before the pinned batched RPA source; "
            "refusing to benchmark an ambiguous baseline.")

    parts = BATCHED_RPA_PACKAGE.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        package = types.ModuleType(name)
        path = (module_root if depth == len(parts)
                else module_root.parents[len(parts) - depth - 1])
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package

    # Note: tuned_params only calls init_logger, and tpu_inference/logger.py
    # is a thin wrapper over vllm.logger, so a stdlib logger stands in for it.
    # envs.py imports only the stdlib and loads from the pin.
    logger = types.ModuleType("tpu_inference.logger")
    logger.init_logger = logging.getLogger
    sys.modules[logger.__name__] = logger

    wrapper = importlib.import_module(f"{BATCHED_RPA_PACKAGE}.wrapper")
    configs = importlib.import_module(f"{BATCHED_RPA_PACKAGE}.configs")
    tuned_params = importlib.import_module(
        f"{BATCHED_RPA_PACKAGE}.tuned_params")
    return wrapper, configs, tuned_params
