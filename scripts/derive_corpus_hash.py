#!/usr/bin/env python3
"""Reference chunked-corpus hash for scripts/corpus_hash.c comparison.

usage: derive_corpus_hash.py KIND MODEL CORPUS [CHUNK-MIB]
  KIND: tokenizers | sentencepiece | tiktoken
  MODEL: tokenizer.json path | .model path | encoding name

Chunks the corpus at the last space boundary of each CHUNK (mirroring
corpus_hash.c), encodes each chunk with the reference library, and
prints count=<n> fnv=<hex> using the same FNV-1a over the id stream.
"""
import sys

CHUNK = 1 << 20

def fnv_add(hv, ids):
    for i in ids:
        hv ^= i
        hv = (hv * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return hv

def main():
    kind, model, path = sys.argv[1], sys.argv[2], sys.argv[3]
    chunk = int(sys.argv[4]) << 20 if len(sys.argv) > 4 else CHUNK
    enc = None
    if kind == "kimi":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        ranks = load_tiktoken_bpe(model)
        pat = "|".join([
            r"[\p{Han}]+",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"\s*[\r\n]+",
            r"\s+(?!\S)",
            r"\s+",
        ])
        tk = tiktoken.Encoding(name="kimi", pat_str=pat,
                               mergeable_ranks=ranks, special_tokens={})
        enc = lambda b: tk.encode(b.decode("utf-8", "replace"))
    elif kind == "tokenizers":
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(model)
        enc = lambda b: t.encode(b.decode("utf-8"), add_special_tokens=False).ids
    elif kind == "sentencepiece":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model)
        enc = lambda b: sp.encode(b)
    else:
        import tiktoken
        tk = tiktoken.get_encoding(model)
        enc = lambda b: tk.encode(b.decode("utf-8"), disallowed_special=())
    hv = 1469598103934665603
    total = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            buf = tail + data
            cut = buf.rfind(b" ")
            if cut < 0 or len(data) < chunk:
                cut = len(buf)
            ids = enc(buf[:cut])
            total += len(ids)
            hv = fnv_add(hv, ids)
            tail = buf[cut:]
    if tail:
        ids = enc(tail)
        total += len(ids)
        hv = fnv_add(hv, ids)
    print("count=%d fnv=%x" % (total, hv))

if __name__ == "__main__":
    main()
