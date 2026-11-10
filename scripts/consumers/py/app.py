import mmap
import os
import sys
import time

from tokensim import tokenizer, default_threads, fold_doc, bytes_equal

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


def rss_kb():
    with open("/proc/self/statm") as f:
        parts = f.read().split()
    return int(parts[1]) * 4


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
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    tok = tokenizer(model, fmt, scheme)
    print(f"threads(auto): {default_threads()}")

    fsz = os.path.getsize(corpus)
    h = FNV0
    total = 0
    checked = 0
    bad = 0
    docs = 0
    rss = [rss_kb(), -1, -1, -1, -1]
    enc_t = 0.0
    dec_t = 0.0
    ebuf = bytearray(doc_bytes * 4 + 64)
    dbuf = bytearray(doc_bytes + 64)
    t0 = time.perf_counter()
    with open(corpus, "rb") as f, mmap.mmap(f.fileno(), 0,
                                            access=mmap.ACCESS_READ) as mm:
        mm.madvise(mmap.MADV_SEQUENTIAL)
        mv = memoryview(mm)
        for off in range(0, fsz, doc_bytes):
            n = min(fsz - off, doc_bytes)
            a = time.perf_counter()
            n_ids = (tok.encode_threads_into(mv[off:off + n], ebuf, threads)
                     if threads else
                     tok.encode_threads_into(mv[off:off + n], ebuf, 0))
            b = time.perf_counter()
            enc_t += b - a
            ids_view = memoryview(ebuf)[:n_ids * 4]
            h = fold_doc(h, ids_view)
            total += n_ids
            docs += 1
            c = time.perf_counter()
            got = tok.decode_into(ids_view, dbuf)
            dec_t += time.perf_counter() - c
            checked += 1
            if not bytes_equal(dbuf[:got], mm[off:off + n]):
                bad += 1
            pos = off + n
            for q in range(1, 5):
                if rss[q] < 0 and pos * 4 >= fsz * q:
                    rss[q] = rss_kb()
        mv.release()
    t1 = time.perf_counter()
    for i in range(5):
        if rss[i] < 0:
            rss[i] = rss_kb()
    print(f"input: {fsz} bytes in {docs} docs of {doc_bytes} bytes")
    print(f"ids: {total}")
    print(f"hash: 0x{h:016x}")
    print(f"roundtrip: {'ok' if bad == 0 else 'MISMATCH'} ({checked} checked)")
    print(f"encode: {enc_t:.3f} s = {fsz / 1e6 / enc_t:.1f} MB/s")
    print(f"decode: {dec_t:.3f} s = {fsz / 1e6 / dec_t:.1f} MB/s")
    print(f"docs (round trip): {t1 - t0:.3f} s = "
          f"{fsz / 1e6 / (t1 - t0):.1f} MB/s")
    print(f"rss: {rss[0]} {rss[1]} {rss[2]} {rss[3]} {rss[4]} KB")

    ph = FNV0
    pids = 0
    p0 = time.perf_counter()
    batch_n = 256
    done = 0
    while done < prompt_calls:
        n = min(prompt_calls - done, batch_n)
        chunk = [PROMPTS[(done + i) % 8].encode() for i in range(n)]
        batch = tok.encode_batch(chunk, threads)
        for raw in batch:
            count = len(raw) // 4
            ph = fold_count(ph, count)
            ph = fold(ph, raw)
            pids += count
        done += n
    p1 = time.perf_counter()
    print(
        f"prompts: {prompt_calls} calls in {p1 - p0:.3f} s = "
        f"{prompt_calls / (p1 - p0):.0f} calls/s, {pids} ids, "
        f"hash 0x{ph:016x}"
    )


if __name__ == "__main__":
    main()
