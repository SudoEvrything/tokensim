import mmap
import os
import sys
import time

from tokensim import tokenizer, default_threads, fold_doc

PRIME = 0x100000001B3
MASK = (1 << 64) - 1
FNV0 = 0xCBF29CE484222325

PROMPTS = [
    "Summarize the following article in one sentence.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "The quick brown fox jumps over the lazy dog.",
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Explain the difference between a list and a tuple in Python.",
    "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY 2 DESC;",
    "Machine learning models are trained on large corpora of text.",
    "The recipe calls for 2 cups of flour, 1 tsp of salt, and 3 eggs.",
]


def fold(h, data):
    for b in data:
        h = ((h ^ b) * PRIME) & MASK
    return h


def fold_count(h, v):
    return fold(h, v.to_bytes(8, "little"))


def main():
    model, fmt, scheme, corpus = sys.argv[1:5]
    doc_bytes = int(sys.argv[5]) if len(sys.argv) > 5 else (16 << 20)
    prompt_calls = int(sys.argv[6]) if len(sys.argv) > 6 else 20000

    tok = tokenizer(model)
    print(f"threads(auto): {default_threads()}")

    fsz = os.path.getsize(corpus)
    h = FNV0
    total = 0
    checked = 0
    bad = 0
    docs = 0
    enc_t = 0.0
    dec_t = 0.0
    t0 = time.perf_counter()
    with open(corpus, "rb") as f, mmap.mmap(f.fileno(), 0,
                                            access=mmap.ACCESS_READ) as mm:
        mm.madvise(mmap.MADV_SEQUENTIAL)
        for off in range(0, fsz, doc_bytes):
            n = min(fsz - off, doc_bytes)
            doc = mm[off:off + n]
            a = time.perf_counter()
            ids = tok.encode(doc)
            enc_t += time.perf_counter() - a
            h = fold_doc(h, ids)
            total += len(ids) // 4
            docs += 1
            c = time.perf_counter()
            back = tok.decode(ids)
            dec_t += time.perf_counter() - c
            checked += 1
            if back != doc:
                bad += 1
    t1 = time.perf_counter()
    print(f"input: {fsz} bytes in {docs} docs of {doc_bytes} bytes")
    print(f"ids: {total}")
    print(f"hash: 0x{h:016x}")
    print(f"roundtrip: {'ok' if bad == 0 else 'MISMATCH'} ({checked} checked)")
    print(f"encode: {enc_t:.3f} s = {fsz / 1e6 / enc_t:.1f} MB/s")
    print(f"decode: {dec_t:.3f} s = {fsz / 1e6 / dec_t:.1f} MB/s")
    print(f"docs (round trip): {t1 - t0:.3f} s = "
          f"{fsz / 1e6 / (t1 - t0):.1f} MB/s")

    ph = FNV0
    pids = 0
    p0 = time.perf_counter()
    batch = tok.encode_batch([PROMPTS[i % 8].encode()
                              for i in range(prompt_calls)])
    for raw in batch:
        count = len(raw) // 4
        ph = fold_count(ph, count)
        ph = fold(ph, raw)
        pids += count
    p1 = time.perf_counter()
    print(
        f"prompts: {prompt_calls} calls in {p1 - p0:.3f} s = "
        f"{prompt_calls / (p1 - p0):.0f} calls/s, {pids} ids, "
        f"hash 0x{ph:016x}"
    )


if __name__ == "__main__":
    main()
