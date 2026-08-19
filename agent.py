#!/usr/bin/env python3
"""Offline local-LLM agent.

Usage:
    python3 agent.py <folder> <prompt>

Reads text files under <folder>, picks the pieces most relevant to
<prompt>, and asks a local GGUF model (via llama-cpp-python) to answer.
Everything after the model file is downloaded once runs fully offline --
no Ollama, no external API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localllm.context import build_context
from localllm.model import load_model, resolve_model_path

SYSTEM_PROMPT = (
    "You are a helpful local coding/document assistant. Answer the user's "
    "prompt using the provided file context when it is relevant. If the "
    "context does not contain the answer, say so instead of guessing."
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fully offline local LLM agent over a folder of files."
    )
    parser.add_argument("folder", help="Folder whose files provide context")
    parser.add_argument("prompt", help="Prompt/question to ask the model")
    parser.add_argument("--model", help="Path to a GGUF model file", default=None)
    parser.add_argument("--ctx-size", type=int, default=4096, help="Model context window in tokens")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads to use")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU (Metal on Apple Silicon, CUDA/ROCm elsewhere); "
        "-1 offloads all layers (default), 0 forces CPU-only",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Max characters of file context to include in the prompt",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: folder not found: {folder}", file=sys.stderr)
        return 1

    context = build_context(folder, args.prompt, args.max_context_chars)

    model_path = resolve_model_path(args.model)
    llm = load_model(
        model_path,
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        verbose=args.verbose,
    )

    user_content = args.prompt
    if context:
        user_content = (
            f"Context from files under {folder}:\n\n{context}\n\n"
            f"Prompt: {args.prompt}"
        )
    else:
        print(f"(no relevant text files found under {folder}; asking without file context)", file=sys.stderr)

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(response["choices"][0]["message"]["content"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
