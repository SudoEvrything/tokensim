#!/usr/bin/env python3
"""Run 146: the differential charsmap check.

The reference exposes Normalize() - the exact normalizer output. This
script compares it against a Python model of OUR normalizer (the rule
set from gen_spm_seqmap.py + the collapse/pending/dummy/escape logic
of src/metaspace.c) over the whole codepoint space, then over random
sequences. Any mismatch is a missing rule or a semantics difference.

Memory-safe: the loop is streaming; only the counters and the first
mismatches are kept.

usage: spm_norm_probe.py [--model PATH] [--max-cps N] [--seqs N]
"""
import random
import sys
import unicodedata

import sentencepiece as spm

sys.path.insert(0, "scripts")
from gen_spm_seqmap import build_map

MARK = "\u2581"
MAXKEY = 4

SPACE_EXTRA = set(range(0x200B, 0x2010)) | {0xFEFF, 0xFFFD, 0x2581}

def is_space_cp(cp):
    ch = chr(cp)
    if ch in " \t\n\v\f\r":
        return True
    if cp == 0x85:
        return False
    return unicodedata.category(ch) in ("Zs", "Zl", "Zp") or cp in SPACE_EXTRA

def model_normalize(text, rules, dummy_prefix=True, remove_extra=True):
    """A faithful model of tsimd_spm_normalize (remove_extra path)."""
    out = []            # the content bytes/chars (as a list of chars)
    pending = False
    have_content = False
    i = 0
    n = len(text)
    while i < n:

        key = None
        for ln in range(min(MAXKEY, n - i), 0, -1):
            k = tuple(ord(c) for c in text[i:i + ln])
            if k in rules:
                key = (k, rules[k])
                break
        if key is None:
            cp = ord(text[i])
            if is_space_cp(cp):
                pending = True
            else:
                if pending and have_content:
                    out.append(" ")
                pending = False
                have_content = True
                out.append(text[i])
            i += 1
            continue
        k, t = key
        if not t:
            i += len(k)      # a deletion rule: emit nothing
            continue
        tgt = "".join(chr(c) for c in t)
        maps_to_space = tgt.startswith(" ")
        if not maps_to_space:
            if pending and have_content:
                out.append(" ")
            pending = False
        have_content = True
        if maps_to_space:
            pending = True
            body = tgt[1:]
        else:
            body = tgt
        if body:
            if not maps_to_space:
                out.append(body)
            else:
                if pending and have_content:
                    out.append(" ")
                    pending = False
                out.append(body)
        i += len(k)
    content = "".join(out)
    if not content:
        return ""
    want_dummy = dummy_prefix and not content.startswith(" ")
    spaces = (1 if want_dummy else 0) + content.count(" ")
    res = content.replace(" ", MARK)
    if want_dummy:
        res = MARK + res
    return res

def main():
    args = sys.argv[1:]
    model = "models/reference/t5_spiece.model"
    max_cps = 0x110000
    seqs = 0
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--max-cps" in args:
        max_cps = int(args[args.index("--max-cps") + 1])
    if "--seqs" in args:
        seqs = int(args[args.index("--seqs") + 1])
    rules = build_map()
    m = spm.SentencePieceProcessor(model_file=model)
    bad = 0
    shown = 0
    for cp in range(1, max_cps):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        ref = m.Normalize(ch)
        mine = model_normalize(ch, rules)
        if ref != mine:
            bad += 1
            if shown < 25:
                shown += 1
                print("U+%04X ref=%s mine=%s rule=%s" % (
                    cp, ref.encode("unicode_escape").decode(),
                    mine.encode("unicode_escape").decode(),
                    " ".join("%X" % c for c in rules.get((cp,), ()))))
    print("single-cp: %d mismatches / %d checked" % (bad, max_cps - 1))
    if seqs:
        rng = random.Random(146)
        pool = []
        for k in list(rules):
            pool.append("".join(chr(c) for c in k))

        bases = [chr(c) for c in range(0x61, 0x7B)] + \
                [chr(c) for c in (0xC0, 0xE9, 0x3B1, 0x430, 0x4E00)]
        marks = [chr(c) for c in (0x300, 0x301, 0x302, 0x303, 0x304, 0x308,
                                  0x30A, 0x327, 0x328, 0x30C, 0x323)]
        spacey = [" ", "\u00a0", "\u200b", "\u3000", "\t"]
        bad2 = 0
        shown2 = 0
        for it in range(seqs):
            kind = rng.randrange(4)
            if kind == 0 and pool:
                s = rng.choice(pool)
            elif kind == 1:
                s = "".join(rng.choice(bases + marks + spacey)
                            for _ in range(rng.randrange(1, 8)))
            elif kind == 2:
                s = "".join(rng.choice(bases) + rng.choice(marks)
                            for _ in range(rng.randrange(1, 4)))
            else:
                s = "".join(rng.choice(bases) + rng.choice(marks) +
                            rng.choice(spacey)
                            for _ in range(rng.randrange(1, 4)))
            ref = m.Normalize(s)
            mine = model_normalize(s, rules)
            if ref != mine:
                bad2 += 1
                if shown2 < 25:
                    shown2 += 1
                    print("SEQ %s ref=%s mine=%s" % (
                        s.encode("unicode_escape").decode(),
                        ref.encode("unicode_escape").decode(),
                        mine.encode("unicode_escape").decode()))
        print("sequences: %d mismatches / %d" % (bad2, seqs))

if __name__ == "__main__":
    main()
