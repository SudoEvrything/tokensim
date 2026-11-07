#!/usr/bin/env python3
"""The tokenizer.json family parity harness (run 137): compare our
encoder against the HuggingFace `tokenizers` reference ON THIS MACHINE,
for the model families the campaign had not yet covered end to end
(llama3, qwen2.5, qwen3, glm4, smollm2, deepseek, nemotron, gemma4).

Both sides encode the same slice with add_special_tokens off. The
reference id list is streamed to a temp file, then both id files are
compared position by position (no id lists in memory at 128-MiB scale).

usage: tokjson_parity.py MODEL [--corpus F] [--limit-mib N]
                             [--probe PATH]
"""
import os
import subprocess
import sys
import tempfile

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: tokjson_parity.py MODEL [--corpus F] [--limit-mib N]")
    model = sys.argv[1]
    corpus = "testdata/owt.txt"
    limit_mib = 2
    probe = "build/encode_probe"
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--corpus":
            corpus = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--limit-mib":
            limit_mib = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--probe":
            probe = sys.argv[i + 1]; i += 2
        else:
            sys.exit("bad arg: " + sys.argv[i])

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(model)
    with open(corpus, "rb") as f:
        data = f.read(limit_mib << 20)
    text = data.decode("utf-8", "replace")
    ref_ids = tk.encode(text, add_special_tokens=False).ids
    with tempfile.NamedTemporaryFile("w", delete=False) as rf:
        for x in ref_ids:
            rf.write("%d\n" % x)
        ref_path = rf.name
    del ref_ids

    ours_path = tempfile.mktemp()
    with open(ours_path, "wb") as of:
        p = subprocess.run([probe, "tokenizer_json", model, "-", "@stdin"],
                           input=data, stdout=of, stderr=subprocess.PIPE)
        if p.returncode != 0:
            os.unlink(ref_path)
            sys.exit("probe failed: " + p.stderr.decode()[:400])

    n_ref = n_ours = diff_positions = 0
    firsts = []
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
            if lr.strip() != lo.strip():
                diff_positions += 1
                if len(firsts) < 8:
                    firsts.append((n_ours - 1, lr.strip(), lo.strip()))
    os.unlink(ref_path)
    os.unlink(ours_path)

    print("model: %s" % model)
    print("ref ids: %d   ours ids: %d" % (n_ref, n_ours))
    print("stream-diff positions: %d" % diff_positions)
    for pos, r, o in firsts:
        print("  near idx %d: ref %s | ours %s" % (pos, r, o))
    if diff_positions == 0 and n_ref == n_ours:
        print("PARITY: EXACT")
        return 0
    print("PARITY: MISMATCH")
    return 1

if __name__ == "__main__":
    sys.exit(main())
