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
   and runs inference in-process, offloading to GPU automatically when the
   build supports it (Metal on Apple Silicon, CUDA/ROCm elsewhere) and
   falling back to CPU otherwise.
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

**macOS (Apple Silicon):** the official PyPI wheel is built with Metal
support already enabled, so `pip install -r requirements.txt` is all you
need -- no `CMAKE_ARGS` or extra flags. `agent.py` offloads all layers to
the GPU by default (`--gpu-layers -1`), so it uses the Metal GPU
automatically. A 1-3B GGUF model comfortably fits Apple Silicon's unified
memory on a 16GB Mac; pass `--gpu-layers 0` to force CPU-only if you ever
need to compare.

### 2. Get a model file (one-time, needs internet -- do this once, then vendor it)

`agent.py` needs a chat model in GGUF format at `models/model.gguf`. Getting
that file onto disk is the *only* step in this whole project that touches
the network, and it's a one-person, one-time job -- not something every
clone has to repeat. See "Vendoring the model into the repo" below for why
that matters if you or your teammates might be behind a restrictive proxy.

Option A -- use the helper script:

```bash
pip install -r requirements-dev.txt   # adds huggingface_hub
python3 scripts/download_model.py
```

This fetches the default model
(`HuggingFaceTB/SmolLM2-135M-Instruct-GGUF`, Q4_K_M, ~90MB) to
`models/model.gguf`. It's intentionally tiny so it fits under GitHub's
100MB no-LFS limit and can be committed straight into the repo (step below).
For noticeably better answer quality at the cost of a much larger,
LFS-tracked file, pass a bigger model, e.g.:

```bash
python3 scripts/download_model.py \
  --repo Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --filename qwen2.5-1.5b-instruct-q4_k_m.gguf
```

(any GGUF chat model works -- Llama 3, Mistral, Phi, Gemma GGUF builds too)

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

### 2b. Vendoring the model into the repo (recommended)

To make `git clone` alone enough to run offline -- no separate download
step on every machine -- commit the model file into the repo. `models/`
is git-ignored *except* for `models/model.gguf` specifically, so the
default (~90MB) model can be committed directly, no Git LFS required:

```bash
python3 scripts/download_model.py        # fetches models/model.gguf
git add models/model.gguf
git commit -m "Vendor local model weights"
git push
```

`scripts/download_model.py` prints these exact commands (and switches to
Git LFS instructions automatically) based on the downloaded file's size.

If you swap in a larger model and still want it vendored, track it with
Git LFS instead (mind GitHub's free-tier LFS quota: 1GB storage / 1GB
bandwidth per month):

```bash
git lfs install
git lfs track "*.gguf"
git add .gitattributes models/model.gguf
git commit -m "Vendor local model weights (LFS)"
git push
```

This download/commit step has to happen once, from a machine with normal
internet access to Hugging Face (a locked-down corporate network or an
unusually restrictive proxy could block it, but that's not typical for a
personal Mac). After that commit lands, everyone who clones the repo --
including on a network that blocks huggingface.co entirely -- has the
model file already and never needs to fetch anything.

### 3. Run it, fully offline

```bash
python3 agent.py ./my_project "Summarize what this project does"
python3 agent.py ./notes "What did I decide about the database schema?"
```

You can disconnect from the network entirely at this point -- inference
runs locally against the file on disk. To double-check, turn on macOS
Airplane Mode (or disconnect Wi-Fi) and re-run the command above; it will
still work since nothing after step 2 touches the network.

## CLI options

```
python3 agent.py <folder> <prompt> [options]

  --model PATH             Path to a .gguf model (default: models/model.gguf,
                            or $LOCALLLM_MODEL_PATH)
  --ctx-size N              Model context window in tokens (default: 4096)
  --max-tokens N             Max tokens to generate (default: 512)
  --max-context-chars N      Max chars of file context to include (default: 6000)
  --threads N               CPU threads to use (default: let llama.cpp choose)
  --gpu-layers N            Layers to offload to GPU: Metal on Apple Silicon,
                            CUDA/ROCm elsewhere (default: -1, all layers;
                            use 0 to force CPU-only)
  --temperature F           Sampling temperature (default: 0.2)
  -v, --verbose              Verbose llama.cpp logging
```

## Repo layout

```
agent.py                  CLI entrypoint
localllm/context.py       Folder scanning, chunking, keyword-based ranking
localllm/model.py         Local GGUF model loading (llama-cpp-python)
scripts/download_model.py One-time model download helper (needs internet)
models/model.gguf         Vendored model weights (tracked in git; see 2b)
requirements.txt          Runtime dependency (llama-cpp-python only)
requirements-dev.txt      Adds huggingface_hub, for the download script
```
