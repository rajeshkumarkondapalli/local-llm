"""Thin wrapper around llama-cpp-python for fully offline local inference.

No network access is used at any point here: llama_cpp.Llama loads a GGUF
file already present on disk and runs inference in-process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.gguf"


def resolve_model_path(cli_value: str | None) -> Path:
    candidate = cli_value or os.environ.get("LOCALLLM_MODEL_PATH") or str(DEFAULT_MODEL_PATH)
    return Path(candidate).expanduser().resolve()


def load_model(
    model_path: Path,
    n_ctx: int,
    n_threads: int | None,
    n_gpu_layers: int = -1,
    verbose: bool = False,
):
    if not model_path.exists():
        print(
            f"error: model file not found at {model_path}\n\n"
            "Download a GGUF model once (requires internet) with:\n"
            "  python3 scripts/download_model.py\n"
            "or place any compatible .gguf file at models/model.gguf\n"
            "(or point LOCALLLM_MODEL_PATH / --model at it). "
            "After that, agent.py runs fully offline.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "error: llama-cpp-python is not installed.\n"
            "Install dependencies with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # n_gpu_layers=-1 offloads every layer to GPU when the build supports it
    # (Metal on Apple Silicon, CUDA/ROCm elsewhere). The official PyPI wheels
    # for macOS ship with Metal enabled, so this is what actually makes the
    # agent fast on an M-series Mac. On a CPU-only build the flag is simply
    # ignored and inference stays on CPU.
    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
