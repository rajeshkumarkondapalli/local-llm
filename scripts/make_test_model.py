#!/usr/bin/env python3
"""Generate a tiny, structurally-valid GGUF model for testing agent.py.

This is NOT a real trained model -- it's a minimal `llama`-architecture
network with deterministic random weights, sized just large enough
(2 layers, 64-dim embeddings) to satisfy llama.cpp's model loader. It loads
and runs through the exact same code path as a real GGUF (tokenize -> graph
build -> sample -> detokenize), so it's useful for exercising agent.py's
plumbing end-to-end without needing a real download -- but its answers are
gibberish. For actual usable answers, use models/model.gguf (see
"Vendoring the model into the repo" in README.md).

No network access is used; this only needs the `gguf` package (a build-time
dependency of this script only, not of agent.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gguf import GGUFWriter, TokenType

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "models" / "test-model.gguf"

# Tiny dimensions -- just enough for llama.cpp to build a working graph.
N_EMBD = 64
N_LAYER = 2
N_HEAD = 4
N_FF = 256
N_CTX_TRAIN = 512
RMS_EPS = 1e-5
SEED = 0

# Vocab: 3 control tokens + 256 byte-fallback tokens (the standard scheme
# every real SPM-tokenizer GGUF model uses), so any UTF-8 input tokenizes
# successfully even without any real word/subword pieces.
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
BOS_ID, EOS_ID, UNK_ID = 1, 2, 0


def build_vocab():
    tokens = list(SPECIAL_TOKENS)
    scores = [0.0, 0.0, 0.0]
    types = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        scores.append(0.0)
        types.append(TokenType.BYTE)
    return tokens, scores, types


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DEFAULT), help="Output .gguf path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    def rand(*shape):
        return rng.standard_normal(shape, dtype=np.float32) * 0.02

    tokens, scores, types = build_vocab()
    n_vocab = len(tokens)

    writer = GGUFWriter(str(out_path), "llama")
    writer.add_name("tiny-test-llm")
    writer.add_context_length(N_CTX_TRAIN)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_layer_norm_rms_eps(RMS_EPS)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(BOS_ID)
    writer.add_eos_token_id(EOS_ID)
    writer.add_unk_token_id(UNK_ID)

    # token_embd.weight: ne={n_embd, n_vocab} -> numpy shape (n_vocab, n_embd)
    writer.add_tensor("token_embd.weight", rand(n_vocab, N_EMBD))
    # output.weight intentionally omitted: llama.cpp ties it to token_embd
    # when absent, halving the tensors we need to generate.
    writer.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))

    for i in range(N_LAYER):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_k.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_v.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_output.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand(N_EMBD, N_FF))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand(N_FF, N_EMBD))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = out_path.stat().st_size
    print(f"Wrote synthetic test model to {out_path} ({size / 1024:.1f} KB)")
    print(
        "This model has random weights and produces gibberish text -- it only "
        "exists to exercise agent.py's loading/tokenize/generate pipeline. Try:\n"
        f"  python3 agent.py . \"hello\" --model {out_path} --max-tokens 16"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
