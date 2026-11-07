import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tsimd

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

def encode_bytes_check(tok, data):
    import array
    lst = tok.encode(data)
    a = array.array("I")
    a.frombytes(tok.encode_bytes(data))
    p = array.array("I")
    p.frombytes(tok.encode_threads(data, 6))
    return list(a) == lst and list(p) == lst and \
        len(tok.encode_bytes(data)) == 4 * len(lst)

def encode_into_check(tok, data):
    import array
    lst = tok.encode(data)
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
    for name, path in cases:
        if not os.path.exists(path):
            print(f"  SKIP {name}: no model at {path} "
                  "(set TOKENSIMD_ROOT to the repo checkout)")
            continue
        tok = tsimd.load(path)
        try:
            ref = hf_ids(path, TEXTS[0])
        except ImportError:
            print("  SKIP hf reference: tokenizers not installed")
            continue
        mine = tok.encode(TEXTS[0].encode("utf-8"))
        ok = mine == ref
        print(f"{name} 'Hello, world!': {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("  mine:", mine)
            print("  ref :", ref)
            fails += 1
        for text in TEXTS[1:]:
            mine = tok.encode(text.encode("utf-8"))
            back = tok.decode(mine).decode("utf-8")
            if back != text:
                print(f"{name} round-trip FAIL: {text[:40]!r}")
                fails += 1
        big = (TEXTS[0] + " ") * 1000
        mine = tok.encode(big.encode("utf-8"))
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
