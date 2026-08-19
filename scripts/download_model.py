#!/usr/bin/env python3
"""One-time helper to fetch a small GGUF chat model for offline use.

This is the *only* step in this project that touches the network. Run it
once while you have internet access; agent.py itself never makes network
calls. If you already have a .gguf file (from any source, on a USB stick,
etc.), you can skip this script entirely -- just copy it to
models/model.gguf or pass --model /path/to/file.gguf to agent.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="GGUF filename in the repo")
    parser.add_argument(
        "--out",
        default=str(MODELS_DIR / "model.gguf"),
        help="Where to save the model (default: models/model.gguf)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "error: huggingface_hub is not installed.\n"
            "Install it with: pip install -r requirements-dev.txt\n\n"
            "Alternatively, download any .gguf chat model manually (browser, "
            "curl, wget, etc.) from Hugging Face or elsewhere and save it to:\n"
            f"  {args.out}",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.filename} from {args.repo} ...")
    downloaded = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=out_path.parent,
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path != out_path:
        if out_path.exists():
            out_path.unlink()
        downloaded_path.rename(out_path)

    print(f"Model ready at {out_path}")
    print("You can now run agent.py fully offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
