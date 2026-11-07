#!/usr/bin/env python3
"""Generate golden test fixtures.

Files written to tests/fixtures/:
  golden_pretokens_<scheme>.tsv  — escaped input TAB pretoken TAB ...
  golden_tokens_<scheme>.tsv     — escaped input TAB id,id,...

Escaping (same scheme as the C test reader): backslash, \\0, \\t, \\n, \\r.
Pretoken fixtures need only the `regex` module (no network). Token
fixtures need tiktoken and its data files."""
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "tests", "fixtures")

PATTERNS = {
    "gpt2": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    "cl100k": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
}

SAMPLES = [
    "",
    "hello world",
    "Hello, WORLD!!  this_is-C++/code (v1.2.3); x=0b1010 & 0xff",
    "I'm sure they'll've said it's Fred's doggo's bone... isn't it?",
    "  leading and   trailing   spaces   ",
    "line1\nline2\r\nline3\rline4\n\n\n  \t mixed \n\t ws",
    "tabs\tand\x0bvertical\x0cforms",
    "numbers 123456789 42 7 1000000 3.14159",
    "unicode: héllo wörld ñ ü ß Ø",
    "cjk: 你好世界 中文测试 日本語テキスト",
    "emoji: 🚀🎉 👨‍👩‍👧‍👦 family + ❤️ heart",
    "arabic: مرحبا بالعالم",
    "devanagari: नमस्ते दुनिया",
    "cyrillic: Привет мир",
    "greek: Γειά σου Κόσμε",
    "curly “quotes” and ‘apostrophes’ — em—dashes…",
    "<|endoftext|>special<|im_start|>tokens<|im_end|>",
    "a",
    "\n",
    " ",
    "!!!???...,,,;;;:::",
    "\x00\x01\x02 binary \xff\xfe\xfd bytes",
    "é́ combining chars",
    "MixedCASE contraction'S DON'T",
    "  \n  x",
]

def esc(s):
    return (s.replace("\\", "\\\\").replace("\x00", "\\0")
             .replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"))

def gen_pretokens():
    try:
        import regex
    except ImportError:
        print("regex module not available; skipping pretoken fixtures")
        return
    for name, pat in PATTERNS.items():
        path = os.path.join(OUT_DIR, "golden_pretokens_%s.tsv" % name)
        with open(path, "w", encoding="utf-8") as f:
            for s in SAMPLES:
                toks = regex.findall(pat, s)
                f.write(esc(s) + "\t" + "\t".join(esc(t) for t in toks) + "\n")
        print("wrote", path)

def gen_tokens():
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not available; skipping token fixtures")
        return
    for name, enc_name in (("gpt2", "r50k_base"), ("cl100k", "cl100k_base")):
        try:
            enc = tiktoken.get_encoding(enc_name)
        except Exception as e:
            print("could not fetch %s (%s); skipping" % (enc_name, e))
            continue
        path = os.path.join(OUT_DIR, "golden_tokens_%s.tsv" % name)
        with open(path, "w", encoding="utf-8") as f:
            for s in SAMPLES:
                ids = enc.encode_ordinary(s)
                f.write(esc(s) + "\t" + ",".join(map(str, ids)) + "\n")
        print("wrote", path)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    gen_pretokens()
    gen_tokens()
