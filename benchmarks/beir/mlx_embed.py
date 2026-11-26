"""EmbeddingGemma 2, for the benchmark: `embed_beir.py` takes this file's MLX encoder (`build_mlx`).
It can also serve embeddings on a Unix socket, so the model loads once for several callers, and
exits after it's been idle a while (it was the embedder of an application since removed).

It runs the model with MLX when this Python has `mlx` and `tokenizers` (Apple silicon), else with
sentence-transformers and PyTorch. Both give the same vectors (cosine 0.9999); MLX embeds a query
in 4 ms warm and 11-28 ms after a pause, against 19 and 40-95 ms, and documents 3.6x as fast.

    python -I mlx_embed.py --socket PATH [--idle SECONDS]

One JSON request per line, one reply per line:

    in:  {"kind": "query" | "document", "texts": ["...", ...]}
    out: {"dims": 256, "n": 2, "vectors": "<base64: n x dims little-endian float32>", "ms": 12.5}
         or {"error": "..."}
    in:  {"ping": true}     out: {"ready": true, "model": "...", "dims": 256}

Documents should be formatted "title: {title} | text: {text}"; queries get the model's search
prompt here. Vectors are the first 256 dimensions (Matryoshka), normalized.
"""

import argparse
import base64
import json
import os
import socketserver
import sys
import threading
import time

MODEL = "google/embeddinggemma-2"
DIMS = 256
MAX_TOKENS = 512
QUERY_PROMPT = "task: search result | query: "


def load():
    """The model, as a function from texts to normalized embeddings (n x 768), and its backend."""
    try:
        import mlx.core  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return load_torch()
    return load_mlx()


def load_torch():
    import torch
    from sentence_transformers import SentenceTransformer

    # The model card warns float16 silently degrades embeddings: bfloat16 on Metal.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = MAX_TOKENS

    def encode(texts):
        return model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    return encode, f"torch ({device})"


def load_mlx():
    # MLX's streams belong to the thread that made them, and requests arrive on a thread each:
    # one worker thread loads and runs the model.
    from concurrent.futures import ThreadPoolExecutor

    worker = ThreadPoolExecutor(max_workers=1)
    encode = worker.submit(build_mlx).result()
    return (lambda texts: worker.submit(encode, texts).result()), "mlx"


def build_mlx():
    """EmbeddingGemma 2's text encoder in MLX, ported from transformers'
    modeling_embedding_gemma2.py: bidirectional attention on every layer, per-layer embeddings
    gating each block, mean pooling over every token. The sliding window (513) never binds at
    512 tokens, so the only mask is for padding in a batch."""
    import math

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    files = ["config.json", "model.safetensors", "tokenizer.json"]
    try:
        root = snapshot_download(MODEL, allow_patterns=files, local_files_only=True)
    except Exception:
        root = snapshot_download(MODEL, allow_patterns=files)
    cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
    tokenizer = Tokenizer.from_file(os.path.join(root, "tokenizer.json"))
    p = "language_model."
    w = {k[len(p) :]: v for k, v in mx.load(os.path.join(root, "model.safetensors")).items() if k.startswith(p)}
    eps, heads, n = cfg["rms_norm_eps"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    eos = cfg["eos_token_id"]
    # bfloat16 rounding turns sqrt(512) into 22.625, as in transformers
    embed_scale = mx.array(math.sqrt(cfg["hidden_size"]), dtype=mx.bfloat16)
    layers = []
    for i in range(n):
        lw = {k[len(f"layers.{i}.") :]: v for k, v in w.items() if k.startswith(f"layers.{i}.")}
        head_dim = lw["self_attn.q_norm.weight"].shape[0]
        theta = cfg["rope_parameters"][cfg["layer_types"][i]]["rope_theta"]
        layers.append((lw, head_dim, lw["self_attn.k_proj.weight"].shape[0] // head_dim, theta))

    def block(x, ple, mask, lw, head_dim, kv_heads, theta):
        B, L, _ = x.shape
        rms = mx.fast.rms_norm
        h = rms(x, lw["input_layernorm.weight"], eps)
        q = rms((h @ lw["self_attn.q_proj.weight"].T).reshape(B, L, heads, head_dim), lw["self_attn.q_norm.weight"], eps)
        k = rms((h @ lw["self_attn.k_proj.weight"].T).reshape(B, L, kv_heads, head_dim), lw["self_attn.k_norm.weight"], eps)
        v = rms((h @ lw["self_attn.v_proj.weight"].T).reshape(B, L, kv_heads, head_dim), None, eps)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        q = mx.fast.rope(q, head_dim, traditional=False, base=theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=False, base=theta, scale=1.0, offset=0)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, L, -1) @ lw["self_attn.o_proj.weight"].T
        x = x + rms(a, lw["post_attention_layernorm.weight"], eps)
        h = rms(x, lw["pre_feedforward_layernorm.weight"], eps)
        h = (nn.gelu_approx(h @ lw["mlp.gate_proj.weight"].T) * (h @ lw["mlp.up_proj.weight"].T)) @ lw["mlp.down_proj.weight"].T
        x = x + rms(h, lw["post_feedforward_layernorm.weight"], eps)
        h = (nn.gelu_approx(x @ lw["ple_block.per_layer_input_gate.weight"].T) * ple) @ lw["ple_block.per_layer_projection.weight"].T
        x = x + rms(h, lw["ple_block.post_per_layer_input_norm.weight"], eps)
        return x * lw["layer_scalar"]

    def forward(ids, keep):
        B, L = ids.shape
        x = w["embed_tokens.weight"][ids] * embed_scale
        d = cfg["hidden_size_per_layer_input"]
        ple = ((x @ w["ple.per_layer_model_projection.weight"].T) * (cfg["hidden_size"] ** -0.5)).reshape(B, L, n, d)
        ple = mx.fast.rms_norm(ple, w["ple.per_layer_projection_norm.weight"], eps)
        mask = keep[:, None, None, :]  # padding: True attends
        for i, layer in enumerate(layers):
            x = block(x, ple[:, :, i, :], mask, *layer)
        x = mx.fast.rms_norm(x, w["norm.weight"], eps) @ w["embedding_projection.weight"].T
        m = keep.astype(mx.float32)[..., None]
        pooled = (x.astype(mx.float32) * m).sum(axis=1) / m.sum(axis=1)
        return pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)

    forward = mx.compile(forward)

    def encode(texts):
        out = []
        for i in range(0, len(texts), 32):
            # Too long: the first tokens and the end-of-sequence token, as sentence-transformers
            # truncates (cutting at 512 instead moved those vectors to cosine 0.988).
            ids = [e.ids if len(e.ids) <= MAX_TOKENS else e.ids[: MAX_TOKENS - 1] + [eos] for e in tokenizer.encode_batch(texts[i : i + 32])]
            # Lengths rounded up to 32 tokens, so compiled graphs are reused: a new length
            # costs 60-80 ms to trace, and most queries fit in the first 32.
            L = -(-max(len(t) for t in ids) // 32) * 32
            batch = mx.array([t + [0] * (L - len(t)) for t in ids], dtype=mx.int32)
            keep = mx.array([[True] * len(t) + [False] * (L - len(t)) for t in ids])
            out.append(np.array(forward(batch, keep)))
        return np.concatenate(out)

    return encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--idle", type=float, default=1800)
    args = ap.parse_args()

    import numpy as np

    encode, backend = load()
    encode(["warm up"])
    lock = threading.Lock()  # one model: one batch at a time
    last_used = [time.time()]

    def embed(kind, texts):
        if kind == "query":
            texts = [QUERY_PROMPT + t for t in texts]
        # Longest first keeps padding per batch small; put back in order after.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        with lock:
            v = encode([texts[i] for i in order])
        out = np.empty((len(texts), DIMS), dtype="<f4")
        out[order] = v[:, :DIMS]
        out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                last_used[0] = time.time()
                try:
                    req = json.loads(line)
                    if req.get("ping"):
                        reply = {"ready": True, "model": MODEL, "dims": DIMS, "backend": backend}
                    else:
                        t = time.perf_counter()
                        texts = [str(x) for x in req["texts"]]
                        v = embed(req.get("kind", "document"), texts) if texts else np.zeros((0, DIMS), dtype="<f4")
                        reply = {"dims": DIMS, "n": len(texts), "vectors": base64.b64encode(v.tobytes()).decode(), "ms": round((time.perf_counter() - t) * 1000, 1)}
                except Exception as e:  # keep serving on bad input
                    reply = {"error": f"{type(e).__name__}: {e}"}
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()
                last_used[0] = time.time()

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    tmp = args.socket + f".{os.getpid()}"
    if os.path.exists(tmp):
        os.unlink(tmp)
    server = Server(tmp, Handler)
    os.chmod(tmp, 0o600)
    os.replace(tmp, args.socket)  # only a loaded model answers on the socket
    ours = os.stat(args.socket).st_ino
    print(json.dumps({"ready": True, "model": MODEL, "backend": backend, "pid": os.getpid()}), flush=True)

    def reaper():
        while True:
            time.sleep(15)
            if time.time() - last_used[0] > args.idle:
                try:
                    # Unless another server has taken the socket over since.
                    if os.stat(args.socket).st_ino == ours:
                        os.unlink(args.socket)
                except OSError:
                    pass
                os._exit(0)

    threading.Thread(target=reaper, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
