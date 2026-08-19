# local-llm

A fully offline, Ollama-free local LLM agent in Python. Point it at a folder
and a prompt, and it answers using a local GGUF model that runs entirely
on your machine via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) --
no daemon, no external API calls, no internet required at run time.

```
python3 agent.py <folder> <prompt>
```

## How it works

1. `localllm/context.py` scans `<folder>` for text files, skips VCS/build
   noise (`.git`, `node_modules`, `models`, etc.), chunks the files, and
   ranks chunks by simple keyword overlap against `<prompt>` (no embedding
   model or network call needed).
2. The best-matching chunks (up to a character budget) are stitched into a
   context block.
3. `localllm/model.py` loads a local `.gguf` model with `llama-cpp-python`
   and runs inference in-process on CPU (or GPU, if you build
   `llama-cpp-python` with GPU support).
4. The answer is printed to stdout.

Nothing here talks to Ollama, OpenAI, or any other service.

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`llama-cpp-python` compiles a small C++ inference engine on install; this is
the only thing that needs a compiler toolchain (`build-essential`/Xcode CLI
tools). No GPU is required -- CPU inference works out of the box.

### 2. Get a model file (one-time, needs internet)

`agent.py` needs a chat model in GGUF format at `models/model.gguf`. Getting
that file onto disk is the *only* step in this whole project that touches
the network -- once it's there, everything is offline.

Option A -- use the helper script:

```bash
pip install -r requirements-dev.txt   # adds huggingface_hub
python3 scripts/download_model.py
```

This fetches a small instruction-tuned model
(`Qwen/Qwen2.5-1.5B-Instruct-GGUF`, ~1GB) to `models/model.gguf`. Pick a
different model with `--repo` / `--filename` (any GGUF chat model works,
e.g. Llama 3, Mistral, Phi, Gemma GGUF builds).

Option B -- download manually from any source (browser, `curl`, a USB
stick, another machine) and save the file as `models/model.gguf`, or point
at it directly:

```bash
python3 agent.py <folder> <prompt> --model /path/to/any-model.gguf
```

or via environment variable:

```bash
export LOCALLLM_MODEL_PATH=/path/to/any-model.gguf
```

### 3. Run it, fully offline

```bash
python3 agent.py ./my_project "Summarize what this project does"
python3 agent.py ./notes "What did I decide about the database schema?"
```

You can disconnect from the network entirely at this point -- inference
runs locally against the file on disk.

## CLI options

```
python3 agent.py <folder> <prompt> [options]

  --model PATH             Path to a .gguf model (default: models/model.gguf,
                            or $LOCALLLM_MODEL_PATH)
  --ctx-size N              Model context window in tokens (default: 4096)
  --max-tokens N             Max tokens to generate (default: 512)
  --max-context-chars N      Max chars of file context to include (default: 6000)
  --threads N               CPU threads to use (default: let llama.cpp choose)
  --temperature F           Sampling temperature (default: 0.2)
  -v, --verbose              Verbose llama.cpp logging
```

## Repo layout

```
agent.py                  CLI entrypoint
localllm/context.py       Folder scanning, chunking, keyword-based ranking
localllm/model.py         Local GGUF model loading (llama-cpp-python)
scripts/download_model.py One-time model download helper (needs internet)
models/                   Put your .gguf model file here (git-ignored)
requirements.txt          Runtime dependency (llama-cpp-python only)
requirements-dev.txt      Adds huggingface_hub, for the download script
```
