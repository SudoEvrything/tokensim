#!/usr/bin/env python3
"""The Gemma-4 Metaspace parity harness (run 66): compare our
tokenizer.json path (normalizer Replace(" " -> U+2581) + the whole-chain
codepoint BPE) against the tokenizers 0.23.2 reference ON THIS MACHINE.
Both sides encode the whole input in one chain. The reference's ids are
dumped to a temp file streaming, then both id files are compared
position-exact in a streaming pass.

usage: nemo_parity.py tokenizer.json corpus [--limit-mib N]
"""
import subprocess
import sys
import tempfile
import os

def main():
    path = sys.argv[1]
    corpus = sys.argv[2]
    limit_mib = None
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--limit-mib":
            limit_mib = int(sys.argv[i + 1])
            i += 1
    from tokenizers import Tokenizer

    with open(corpus, "rb") as f:
        data = f.read()
    if limit_mib:
        data = data[: limit_mib << 20]
    text = data.decode("utf-8", "replace")

    tok = Tokenizer.from_file(path)
    ref_ids = tok.encode(text).ids
    with tempfile.NamedTemporaryFile("w", delete=False) as rf:
        for x in ref_ids:
            rf.write("%d\n" % x)
        ref_path = rf.name
    del ref_ids

    ours_path = tempfile.mktemp()
    p = subprocess.run(
        ["./build/tsimd-tokenize", "--format", "tokenizer_json", path, "auto"],
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
