import sys
import os

BUILD = os.environ.get("TOKENSIMD_BUILD_DIR")
if BUILD:
    sys.path.insert(0, BUILD)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tokensim
import tsimd

assert tsimd.version == tokensim.version == "1.0.3"
assert tsimd.Error is tokensim.Error
assert tsimd.Tokenizer is tokensim.Tokenizer

ROOT = os.environ.get("TOKENSIMD_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "héllo wörld — ünïcode tëxt with émojis 🎉 and symbols €£¥",
    "spaces    and\ttabs\nand\r\nnewlines   ",
    "数字123と漢字のテスト、日本語のテキスト。",
    "x" * 500,
]

def hf_ids(model_path, text):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(model_path)
    return t.encode(text, add_special_tokens=False).ids

def ids_of(tok, data):
    import array
    a = array.array("I")
    a.frombytes(tok.encode(data))
    return list(a)

def fast_path_check(tok, data):
    import array
    raw = tok.encode(data)
    if not isinstance(raw, bytes):
        print("encode did not return bytes")
        return False
    lst = tok.encode_list(data)
    a = array.array("I")
    a.frombytes(raw)
    if list(a) != lst:
        print("encode bytes differ from encode_list")
        return False
    call = tok(data)
    if call != raw:
        print("__call__ differs from encode")
        return False
    if tok.encode_bytes(data) != raw:
        print("encode differs from encode_bytes")
        return False
    if tok.encode_threads(data, 1) != raw:
        print("encode_threads(1) differs from encode")
        return False
    if tok.decode(raw) != data:
        print("decode(bytes) round-trip differs")
        return False
    if tok.decode(lst) != data:
        print("decode(list) round-trip differs")
        return False
    try:
        tok.decode(raw[:-2])
        print("decode accepted a ragged buffer")
        return False
    except tokensim.Error:
        pass
    return True

def auto_default_check(tok, data, big):
    if tokensim.default_threads() < 1:
        print("default_threads < 1")
        return False
    if tok.encode(data) != tok.encode_threads(data, 1):
        print("auto default differs from 1t")
        return False
    if tok.encode(big) != tok.encode_threads(big, 1):
        print("auto default differs from 1t on a big payload")
        return False
    if tok.encode(big) != tok.encode_threads(big, 6):
        print("auto default differs from 6t on a big payload")
        return False
    if tok(big) != tok.encode(big):
        print("__call__ differs from encode on a big payload")
        return False
    import array
    a = array.array("I")
    a.frombytes(tok.encode_threads(big))
    if list(a) != tok.encode_list(big):
        print("default encode_threads differs from encode_list on a big payload")
        return False
    return True

def binding_error_check(tok, data, model_path):
    try:
        tok(data, bad=1)
        print("__call__ accepted keyword arguments")
        return False
    except TypeError:
        pass
    try:
        tok.encode("a string not bytes")
        print("encode accepted a str")
        return False
    except TypeError:
        pass
    try:
        tok.decode("not a list or buffer")
        print("decode accepted a str")
        return False
    except TypeError:
        pass
    ids_bytes = tok.encode(data)
    if tok.decode(list(tok.encode_list(data)), True) != tok.decode(ids_bytes,
                                                                   False):
        print("skip_special decode differs on special-free input")
        return False
    try:
        tsimd.load("/nonexistent/model.json")
        print("load accepted a missing model")
        return False
    except tsimd.Error:
        pass
    try:
        tsimd.load(model_path, "nonsense")
        print("load accepted an unknown format")
        return False
    except ValueError:
        pass
    try:
        tsimd.load(model_path, "tiktoken", "nonsense")
        print("load accepted an unknown scheme")
        return False
    except ValueError:
        pass
    try:
        tok.encode_threads_into(data, bytes(64), 6)
        print("encode_threads_into accepted a read-only buffer")
        return False
    except (tsimd.Error, TypeError, BufferError):
        pass
    return True

def batch_check(tok, docs):
    expect = [tok.encode(d) for d in docs]
    got = tok.encode_batch(docs)
    if len(got) != len(docs):
        print("encode_batch lost documents")
        return False
    for a, b in zip(got, expect):
        if a != b:
            print("encode_batch ids differ from the per-doc encode")
            return False
    if tok.encode_batch([]) != []:
        print("encode_batch([]) is not empty")
        return False
    try:
        tok.encode_batch([b"ok", "a str"])
        print("encode_batch accepted a str")
        return False
    except TypeError:
        pass
    joined = b"".join(got)
    if len(joined) % 4 != 0:
        print("encode_batch produced ragged output")
        return False
    return True

def thread_safety_check(tok, big):
    import threading
    expect = tok.encode(big)
    results = []
    lock = threading.Lock()

    def work():
        out = [tok.encode(big) for _ in range(3)]
        with lock:
            results.append(out)

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if len(results) != 4:
        print("thread check lost results")
        return False
    for out in results:
        for raw in out:
            if raw != expect:
                print("concurrent encode ids differ")
                return False
    return True

def encode_bytes_check(tok, data):
    import array
    lst = tok.encode_list(data)
    a = array.array("I")
    a.frombytes(tok.encode_bytes(data))
    p = array.array("I")
    p.frombytes(tok.encode_threads(data, 6))
    return list(a) == lst and list(p) == lst and \
        len(tok.encode_bytes(data)) == 4 * len(lst)

def encode_into_check(tok, data):
    import array
    lst = tok.encode_list(data)
    bound = tok.encode_bound(len(data))
    if bound < 4 * len(lst):
        print(f"encode_bound suspicious: {bound} for {len(lst)} ids")
        return False
    buf = bytearray((bound + 3) & ~3)
    n = tok.encode_threads_into(data, buf, 6)
    a = array.array("I")
    a.frombytes(bytes(buf[: 4 * n]))
    if list(a) != lst:
        print("encode_threads_into ids differ")
        return False
    n1 = tok.encode_threads_into(data, buf, 1)
    a1 = array.array("I")
    a1.frombytes(bytes(buf[: 4 * n1]))
    if list(a1) != lst:
        print("encode_threads_into(1t) ids differ")
        return False
    small = bytearray(4)
    try:
        tok.encode_threads_into(data, small, 6)
        print("encode_threads_into accepted an undersized buffer")
        return False
    except tsimd.Error:
        pass
    ba = bytearray(8)
    try:
        tok.encode_threads_into(data, ba, 6)
        print("encode_threads_into accepted a bytearray of 8 bytes")
        return False
    except tsimd.Error:
        pass
    mv = memoryview(buf)
    n2 = tok.encode_threads_into(data, buf, 6)
    if n2 != n:
        print("encode_threads_into not repeatable")
        return False
    return True

def main():
    fails = 0
    cases = [
        ("deepseek", ROOT + "/models/reference/deepseek_tokenizer.json"),
        ("gemma4", ROOT + "/models/reference/gemma4_tokenizer.json"),
        ("qwen2.5", ROOT + "/models/reference/qwen2.5-0.5b_tokenizer.json"),
    ]
    if not hasattr(tokensim, "tokenizer"):
        print("tokensim.tokenizer alias missing")
        fails += 1
    for name, path in cases:
        if not os.path.exists(path):
            print(f"  SKIP {name}: no model at {path} "
                  "(set TOKENSIMD_ROOT to the repo checkout)")
            continue
        tok = tsimd.load(path)
        tok2 = tokensim.tokenizer(path)
        if type(tok2) is not type(tok):
            print("tokensim.tokenizer did not return a Tokenizer")
            fails += 1
        try:
            ref = hf_ids(path, TEXTS[0])
        except ImportError:
            print("  SKIP hf reference: tokenizers not installed")
            continue
        mine = ids_of(tok, TEXTS[0].encode("utf-8"))
        ok = mine == ref
        print(f"{name} 'Hello, world!': {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("  mine:", mine)
            print("  ref :", ref)
            fails += 1
        for text in TEXTS[1:]:
            mine = ids_of(tok, text.encode("utf-8"))
            back = tok.decode(mine).decode("utf-8")
            if back != text:
                print(f"{name} round-trip FAIL: {text[:40]!r}")
                fails += 1
        big = (TEXTS[0] + " ") * 1000
        mine = ids_of(tok, big.encode("utf-8"))
        if len(mine) < 1000:
            print(f"{name} big-encode suspicious: {len(mine)} ids")
            fails += 1
    if fails:
        print(f"test_tsimd: {fails} fails")
        return 1
    if not os.path.exists(ROOT + "/models/reference/deepseek_tokenizer.json"):
        print("  model-dependent checks skipped (no models dir; "
              "set TOKENSIMD_ROOT to the repo checkout)")
        print("test_tsimd: OK (portable checks only)")
        return 0
    tok_b = tsimd.load(ROOT + "/models/reference/deepseek_tokenizer.json")
    if fast_path_check(tok_b, TEXTS[0].encode("utf-8") * 40):
        print("fast-path default OK")
    else:
        fails += 1
        print("fast-path default FAIL")
    if auto_default_check(tok_b, TEXTS[0].encode("utf-8") * 40,
                          (TEXTS[0] + " ") .encode("utf-8") * 20000):
        print("automatic-thread default OK")
    else:
        fails += 1
        print("automatic-thread default FAIL")
    if binding_error_check(tok_b, TEXTS[0].encode("utf-8") * 40,
                           ROOT + "/models/reference/deepseek_tokenizer.json"):
        print("binding error paths OK")
    else:
        fails += 1
        print("binding error paths FAIL")
    if thread_safety_check(tok_b, (TEXTS[0] + " ").encode("utf-8") * 20000):
        print("concurrent encode thread safety OK")
    else:
        fails += 1
        print("concurrent encode thread safety FAIL")
    if batch_check(tok_b, [t.encode("utf-8") for t in TEXTS]):
        print("encode_batch OK")
    else:
        fails += 1
        print("encode_batch FAIL")
    if encode_bytes_check(tok_b, TEXTS[0].encode("utf-8") * 40):
        print("encode_bytes/threads parity OK")
    else:
        fails += 1
        print("encode_bytes/threads parity FAIL")
    if encode_into_check(tok_b, TEXTS[0].encode("utf-8") * 40):
        print("encode_threads_into parity OK")
    else:
        fails += 1
        print("encode_threads_into parity FAIL")
    if fails:
        print(f"test_tsimd: {fails} fails")
        return 1
    print("test_tsimd: all cases pass")
    return 0

if __name__ == "__main__":
    sys.exit(main())
