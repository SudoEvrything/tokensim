#!/usr/bin/env python3
"""The sentencepiece-BPE parity harness (run 66): compare our
metaspace + codepoint-init BPE against the sentencepiece reference ON
THIS MACHINE. The whole input is ONE symbol chain on both sides (the
per-line split is NOT equivalent — the leading marks change the
folds). Memory-safe at the 20 MiB scale: the reference's id list is
dumped to a temp file streaming, then both id files are compared
position-exact in a streaming pass (no id lists in memory).

usage: spm_parity.py model corpus [--limit-mib N] [--probe PATH]
"""
import subprocess
import sys
import tempfile
import os

def main():
    model = sys.argv[1]
    corpus = sys.argv[2]
    limit_mib = None
    probe = "build/spm_probe"
    cli = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--limit-mib":
            limit_mib = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--probe":
            probe = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cli":
            cli = sys.argv[i + 1]
            i += 2
        else:
            sys.exit("bad arg: " + sys.argv[i])
    import sentencepiece as spm

    with open(corpus, "rb") as f:
        data = f.read()
    if limit_mib:
        data = data[: limit_mib << 20]
    taken = len(data)
    text = data.decode("utf-8", "replace")

    m = spm.SentencePieceProcessor(model_file=model)
    ref_ids = m.encode(text, out_type=int)
    with tempfile.NamedTemporaryFile("w", delete=False) as rf:
        for x in ref_ids:
            rf.write("%d\n" % x)
        ref_path = rf.name
    del ref_ids

    ours_path = tempfile.mktemp()
    if cli:
        p = subprocess.run(
            [cli, "--format", "sentencepiece", model, "spm"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p.returncode != 0:
            os.unlink(ref_path)
            sys.exit("cli failed")

        with open(ours_path, "w") as of:
            for x in p.stdout.decode().strip().split():
                of.write(x + "\n")
    else:
        with open(ours_path, "wb") as of:
            p = subprocess.run(
                [probe, model, "@stdin"],
                input=data, stdout=of, stderr=subprocess.PIPE)
        if p.returncode != 0:
            os.unlink(ref_path)
            sys.exit("probe failed: " + p.stderr.decode())

    n_ref = n_ours = 0
    mismatches = []
    with open(ref_path) as rf, open(ours_path) as of:
        while True:
            lr = rf.readline()
            lo = of.readline()
            if not lr and not lo:
                break
            if lr:
                n_ref += 1
            if lo:
                n_ours += 1
            if lr.strip() != lo.strip() and len(mismatches) < 20:
                mismatches.append(
                    (n_ours - 1, lr.strip(), lo.strip()))
    n_total = n_ref
    bad = n_ref - n_ours if n_ref >= n_ours else n_ours - n_ref

    diff_positions = 0
    with open(ref_path) as rf, open(ours_path) as of:
        for a, b in zip(rf, of):
            if a != b:
                diff_positions += 1

    with open(ref_path) as rf, open(ours_path) as of:
        n_r = sum(1 for _ in rf)
        n_o = sum(1 for _ in of)
    tail = abs(n_r - n_o)

    print("bytes fed: %d" % taken)
    print("ref ids : %d   ours ids: %d" % (n_r, n_o))
    print("stream-diff positions: %d (tail length delta %d)"
          % (diff_positions, tail))
    if mismatches:
        for pos, r, o in mismatches[:10]:
            print("  first diffs near idx %d: ref %s | ours %s" % (pos, r, o))
    os.unlink(ref_path)
    os.unlink(ours_path)
    if diff_positions == 0 and tail == 0:
        print("PARITY: EXACT")
        return 0
    print("PARITY: MISMATCH")
    return 1

if __name__ == "__main__":
    sys.exit(main())
