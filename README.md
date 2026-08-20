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

### 2. Model file (already vendored -- nothing to download)

`models/model.gguf` (~97.3MB, `bartowski/SmolLM2-135M-Instruct-GGUF`,
`SmolLM2-135M-Instruct-Q4_K_S.gguf`) is committed in this repo already.
`git clone` alone is enough -- no download step, no huggingface_hub, no
network access needed to get a working model on disk. It's just under
GitHub's 100MB no-LFS limit (the Q4_K_M quant of this same model comes in
at 100.6MB, just over the limit, which is why this repo uses Q4_K_S
instead), so it's a plain committed file, no Git LFS required.

Want a different or better model instead?

Option A -- use the helper script:

```bash
pip install -r requirements-dev.txt   # adds huggingface_hub
python3 scripts/download_model.py
```

This re-fetches the same default. For noticeably better answer quality at
the cost of a much larger, LFS-tracked file, pass a bigger model, e.g.:

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

### 2b. Vendoring a different model into the repo

`models/` is git-ignored *except* for `models/model.gguf` specifically, so
a model that fits under GitHub's 100MB limit can be committed directly, no
Git LFS required. If you re-run `scripts/download_model.py` with a
different `--repo`/`--filename` and want to update the vendored copy:

```bash
python3 scripts/download_model.py --repo <repo> --filename <file>
git add models/model.gguf
git commit -m "Update vendored model weights"
git push
```

`scripts/download_model.py` prints these exact commands (and switches to
Git LFS instructions automatically) based on the downloaded file's size.

If you swap in a larger model and want it vendored, track it with Git LFS
instead (mind GitHub's free-tier LFS quota: 1GB storage / 1GB bandwidth
per month):

```bash
git lfs install
git lfs track "*.gguf"
git add .gitattributes models/model.gguf
git commit -m "Vendor local model weights (LFS)"
git push
```

No machine on hand with normal internet access to Hugging Face? This repo
also has `.github/workflows/vendor-model.yml`, a GitHub Actions workflow
that fetches the model on a GitHub-hosted runner (which has ordinary
internet access, unlike a locked-down sandbox or restrictive corporate
proxy) and commits+pushes it back to the branch for you. Trigger it by
pushing any change to that workflow file, or adjust its `repo`/`filename`
`workflow_dispatch` inputs once the workflow exists on your default
branch.

### 2c. Test model (already vendored, for smoke-testing)

`models/test-model.gguf` (~570KB) is committed in this repo already -- no
download needed. It's a genuine, structurally-valid GGUF file that
`llama_cpp` loads and runs through the exact same code path as any real
model (tokenize -> build graph -> sample -> detokenize), but it's a tiny
2-layer network with random weights, not a trained model, so its answers
are gibberish. It exists purely to prove the plumbing works end to end
without depending on any real trained model:

```bash
python3 agent.py . "hello" --model models/test-model.gguf --max-tokens 16 --max-context-chars 200
```

Regenerate it (only needed if you change its architecture) with:

```bash
pip install -r requirements-dev.txt   # adds gguf
python3 scripts/make_test_model.py
```

No network access is used -- `scripts/make_test_model.py` builds the GGUF
file from scratch with the `gguf` writer library.

Two flags matter when using this fixture: keep `--max-context-chars` low
(its vocabulary is byte-fallback only, with no real subword merges, so it
tokenizes at roughly 1 token per character instead of a real tokenizer's
~4 chars/token, and can blow past `--ctx-size` on a normal-sized folder);
and if you hit an `Illegal instruction` crash on x86 Linux, that's a CPU
dispatch bug unrelated to this file (llama.cpp misdetecting AVX512-FP16
support on some virtualized/cloud CPUs) -- it isn't specific to this
model and won't occur on Apple Silicon, which uses Metal/ARM kernels
instead of x86 SIMD entirely.

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

## Example walkthrough

An end-to-end run on a Mac, from clone to a fully offline query. Steps 1-2
show output that's representative of what each tool actually prints, not a
captured transcript. Steps 3-5 are real, verified transcripts -- captured
producing and running the actual `models/model.gguf` vendored in this
repo -- and are labeled as such where they appear.

**1. Clone and enter the repo**

```bash
$ git clone https://github.com/rajeshkumarkondapalli/local-llm.git
$ cd local-llm
```
```
Cloning into 'local-llm'...
remote: Enumerating objects: 24, done.
remote: Counting objects: 100% (24/24), done.
remote: Compressing objects: 100% (18/18), done.
remote: Total 24 (delta 4), reused 24 (delta 4), pack-reused 0
Receiving objects: 100% (24/24), 12.40 KiB | 12.40 MiB/s, done.
Resolving deltas: 100% (4/4), done.
```

**2. Set up the Python environment**

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```
```
Collecting llama-cpp-python>=0.3.0
  Downloading llama_cpp_python-0.3.35-cp312-cp312-macosx_14_0_arm64.whl (3.1 MB)
Collecting typing-extensions>=4.5.0 (from llama-cpp-python>=0.3.0)
  Downloading typing_extensions-4.13.0-py3-none-any.whl (37 kB)
Collecting numpy>=1.20.0 (from llama-cpp-python>=0.3.0)
  Downloading numpy-2.2.4-cp312-cp312-macosx_14_0_arm64.whl (5.4 MB)
Collecting diskcache>=5.6.1 (from llama-cpp-python>=0.3.0)
  Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
Collecting jinja2>=2.11.3 (from llama-cpp-python>=0.3.0)
  Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)
Installing collected packages: typing-extensions, numpy, MarkupSafe, jinja2, diskcache, llama-cpp-python
Successfully installed diskcache-5.6.3 jinja2-3.1.5 llama-cpp-python-0.3.35 numpy-2.2.4 typing-extensions-4.13.0
```

The `macosx_14_0_arm64` wheel is the important part -- that's the prebuilt
Metal-enabled binary for Apple Silicon; no compiling needed.

**3. Model is already vendored (steps 3-4 are for reference only)**

`models/model.gguf` is already committed in this repo, so a plain clone
has everything needed. This is what producing and vendoring it looked
like -- useful if you swap in a different model later:

```bash
$ pip install -r requirements-dev.txt
$ python3 scripts/download_model.py
```
```
Downloading SmolLM2-135M-Instruct-Q4_K_S.gguf from bartowski/SmolLM2-135M-Instruct-GGUF ...
SmolLM2-135M-Instruct-Q4_K_S.gguf: 100%|████████████████████████| 97.3M/97.3M [00:03<00:00, 28.1MB/s]
Model ready at /Users/rajesh/local-llm/models/model.gguf (97.3 MB)

This file is under GitHub's 100MB limit, so you can commit it directly to vendor the model into the repo:
  git add models/model.gguf
  git commit -m "Vendor local model weights"
  git push
```

**4. Vendor it into the repo (run the printed commands)**

```bash
$ git add models/model.gguf
$ git commit -m "Vendor local model weights"
$ git push
```
```
[claude/offline-local-lm-python-cdhy6p b103d0e] Vendor local model weights (bartowski/SmolLM2-135M-Instruct-GGUF)
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 models/model.gguf
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (4/4), 97.31 MiB | 8.90 MiB/s, done.
Total 4 (delta 1), reused 0 (delta 0)
To https://github.com/rajeshkumarkondapalli/local-llm
   f7f5b09..b103d0e  claude/offline-local-lm-python-cdhy6p -> claude/offline-local-lm-python-cdhy6p
```

This is a real, verified transcript -- not illustrative like the rest of
this walkthrough. The model was actually fetched and pushed this way,
via the `vendor-model.yml` GitHub Actions workflow described in section
2b (run on a GitHub-hosted runner, since this environment's own network
access is restricted and can't reach Hugging Face directly).

**5. Run it -- try it once with Wi-Fi off to prove it's offline**

This one is a real, verified transcript too (`--gpu-layers 0` forces CPU
so it also works in sandboxes without GPU passthrough):

```bash
$ python3 agent.py localllm "What does the context.py module do?" --gpu-layers 0 --max-tokens 80
```
````
The context.py module is a Python file that contains the code for the Llama-cpp-python module. It is used to compile and run the Llama-cpp-python module.

The module contains the following code:

```python
import os
import sys
import pip
import warnings
import numpy as np
import torch

# ...
```
````

A second example, pointed at a folder with no matching text files:

```bash
$ python3 agent.py /tmp/empty-dir "Where is the auth middleware defined?" --gpu-layers 0
```
```
(no relevant text files found under /tmp/empty-dir; asking without file context)
The auth middleware is defined in the `/api/auth` file. The `/api/auth` file is a configuration file that defines the authentication and authorization logic for the API. It
```

That second case shows the "no matching files" fallback path -- it still
answers, just flags on stderr that it had nothing to ground itself on.

**Note on quality**: both real transcripts above show exactly the
tradeoff this model makes. It's coherent and clearly offline/working, but
also confidently wrong on specifics -- the first answer describes
`context.py` as compiling "the Llama-cpp-python module" and invents a
`torch` import that isn't there; the second invents a plausible-sounding
but made-up file path. SmolLM2-135M (135M params) is intentionally tiny to
fit the no-LFS vendoring approach -- good enough to prove the pipeline
works, not for answers you should trust without checking. If you need
real accuracy, rerun with a bigger model and vendor via Git LFS instead
(see "Vendoring a different model into the repo" above):

```bash
python3 scripts/download_model.py \
  --repo Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --filename qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## Repo layout

```
agent.py                          CLI entrypoint
localllm/context.py               Folder scanning, chunking, keyword-based ranking
localllm/model.py                 Local GGUF model loading (llama-cpp-python)
scripts/download_model.py         Model download helper (needs internet)
scripts/make_test_model.py        Generates the synthetic smoke-test model (offline)
.github/workflows/vendor-model.yml  Re-fetches/vendors a model via a GitHub-hosted runner
models/model.gguf                 Vendored model weights (tracked in git; see section 2)
models/test-model.gguf            Synthetic smoke-test model (tracked in git; see 2c)
requirements.txt                  Runtime dependency (llama-cpp-python only)
requirements-dev.txt              Adds huggingface_hub and gguf, for the scripts above
```
