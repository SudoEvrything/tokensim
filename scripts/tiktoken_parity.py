#!/usr/bin/env python3
"""The tiktoken-format parity harness (run 70): compare our tiktoken
path (the SIMD walker + the batched keyed serve) against the tiktoken
reference ON THIS MACHINE. The reference encodes the whole bounded
slice in one pass (the pretokenization identical on both sides); the
ids are dumped to temp files and compared position-exact in a
streaming pass.

usage: tiktoken_parity.py vocab.tiktoken cl100k|o200k corpus [--limit-mib N]
"""
import subprocess
import sys
import tempfile
import os

CL100K_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
O200K_PAT = r"""\s?(?!\p{L})|\p{Lu}\p{Ll}+|\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # unused; cl100k only

def main():
    path = sys.argv[1]
    scheme = sys.argv[2]
    corpus = sys.argv[3]
    limit_mib = None
    for i in range(4, len(sys.argv)):
        if sys.argv[i] == "--limit-mib":
            limit_mib = int(sys.argv[i + 1])
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    with open(corpus, "rb") as f:
        data = f.read()
    if limit_mib:
        data = data[: limit_mib << 20]
    text = data.decode("utf-8", "replace")

    enc = None
    try:
        enc = tiktoken.get_encoding(scheme + "_base")
    except BaseException:
        enc = None
    if enc is None:
        ranks = load_tiktoken_bpe(path)
        enc = tiktoken.Encoding(
            name="parity_" + scheme,
            pat_str=CL100K_PAT if scheme == "cl100k" else O200K_PAT,
            mergeable_ranks=ranks,
            special_tokens={},
        )
    ref_ids = enc.encode(text, disallowed_special=())
    with tempfile.NamedTemporaryFile("w", delete=False) as rf:
        for x in ref_ids:
            rf.write("%d\n" % x)
        ref_path = rf.name
    del ref_ids

    ours_path = tempfile.mktemp()
    p = subprocess.run(
        ["./build/tsimd-tokenize", "--format", "tiktoken", path, scheme],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        os.unlink(ref_path)
        sys.exit("cli failed")
    with open(ours_path, "w") as of:
        for x in p.stdout.decode().strip().split():
            of.write(x + "\n")

    diff_positions = 0
    with open(ref_path) as rf, open(ours_path) as of:
        for a, b in zip(rf, of):
            if a != b:
                diff_positions += 1
    with open(ref_path) as rf:
        n_r = sum(1 for _ in rf)
    with open(ours_path) as of:
        n_o = sum(1 for _ in of)
    tail = abs(n_r - n_o)
    print("bytes fed: %d  ref ids: %d  ours ids: %d" % (len(data), n_r, n_o))
    print("stream-diff positions: %d (tail delta %d)" % (diff_positions, tail))
    os.unlink(ref_path)
    os.unlink(ours_path)
    if diff_positions == 0 and tail == 0:
        print("PARITY: EXACT")
        return 0
    print("PARITY: MISMATCH")
    return 1

if __name__ == "__main__":
    sys.exit(main())
