#!/usr/bin/env python3
"""Minimal repro of the deepseek full-owt id divergence (run 319).

The reference merges `"` + `S` into one piece when `S` ends the
string after `")・`, ours keeps them separate. Count-preserving,
rare (one seam per ~476 chunks of owt.txt).
"""
import sys
sys.path.insert(0, "bindings/python")
import tsimd
from tokenizers import Tokenizer

t = Tokenizer.from_file("models/reference/deepseek_tokenizer.json")
tok = tsimd.load("models/reference/deepseek_tokenizer.json")
s = '")\u30fb"S'
o = tok.encode(s.encode("utf-8"))
r = t.encode(s, add_special_tokens=False).ids
print("ours:", o, "ref:", r, "MATCH" if o == r else "DIFF")
sys.exit(0 if o == r else 1)
