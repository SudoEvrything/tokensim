#!/usr/bin/env python3
"""Run 146: regenerate the sentencepiece charsmap FAITHFULLY.

The reference normalizer is a greedy longest-match trie over the
precompiled charsmap. The run-66 probe captured only the SINGLE
codepoint rules, so decomposed canonical pairs ("a" + U+0301) never
matched the precomposed piece and diverged (t5: "a\\u0301" gave
3 9 2 against the reference's 3 2975).

This script repeats the reference Builder::BuildMapInternal +
RemoveRedundantMap exactly (sentencepiece-sys-0.14.0/source/src/
builder.cc), using Python's unicodedata as the composer/decomposer.

usage: gen_spm_seqmap.py [--check] [--emit PATH]
  --check  compare the generated SINGLE-codepoint subset against the
           current src/spm_charmap.h (a generator-fidelity gate)
  --emit   write the full rule table as a C header
"""
import sys
import unicodedata

MAX_UNICODE = 0x10FFFF

REF_UNMAPPED = (
    0x32FF, 0xA7F2, 0xA7F3, 0xA7F4, 0xAB69, 0xFF5E, 0x10781, 0x10782,
    0x10783, 0x10784, 0x10785, 0x10787, 0x10788, 0x10789, 0x1078A, 0x1078B,
    0x1078C, 0x1078D, 0x1078E, 0x1078F, 0x10790, 0x10791, 0x10792, 0x10793,
    0x10794, 0x10795, 0x10796, 0x10797, 0x10798, 0x10799, 0x1079A, 0x1079B,
    0x1079C, 0x1079D, 0x1079E, 0x1079F, 0x107A0, 0x107A1, 0x107A2, 0x107A3,
    0x107A4, 0x107A5, 0x107A6, 0x107A7, 0x107A8, 0x107A9, 0x107AA, 0x107AB,
    0x107AC, 0x107AD, 0x107AE, 0x107AF, 0x107B0, 0x107B2, 0x107B3, 0x107B4,
    0x107B5, 0x107B6, 0x107B7, 0x107B8, 0x107B9, 0x107BA, 0x1E030, 0x1E031,
    0x1E032, 0x1E033, 0x1E034, 0x1E035, 0x1E036, 0x1E037, 0x1E038, 0x1E039,
    0x1E03A, 0x1E03B, 0x1E03C, 0x1E03D, 0x1E03E, 0x1E03F, 0x1E040, 0x1E041,
    0x1E042, 0x1E043, 0x1E044, 0x1E045, 0x1E046, 0x1E047, 0x1E048, 0x1E049,
    0x1E04A, 0x1E04B, 0x1E04C, 0x1E04D, 0x1E04E, 0x1E04F, 0x1E050, 0x1E051,
    0x1E052, 0x1E053, 0x1E054, 0x1E055, 0x1E056, 0x1E057, 0x1E058, 0x1E059,
    0x1E05A, 0x1E05B, 0x1E05C, 0x1E05D, 0x1E05E, 0x1E05F, 0x1E060, 0x1E061,
    0x1E062, 0x1E063, 0x1E064, 0x1E065, 0x1E066, 0x1E067, 0x1E068, 0x1E069,
    0x1E06A, 0x1E06B, 0x1E06C, 0x1E06D, 0x1F16C, 0x1F23B, 0x1FBF0, 0x1FBF1,
    0x1FBF2, 0x1FBF3, 0x1FBF4, 0x1FBF5, 0x1FBF6, 0x1FBF7, 0x1FBF8, 0x1FBF9,
)

REF_DELETED = (
    list(range(0x01, 0x09)) + [0x0B] + list(range(0x0E, 0x20)) +
    [0x7F, 0x8F, 0x9F]
)

REF_UNMAPPED_SEQ = ((0x11935, 0x11930),)

def is_unicode_char(cp):
    return 0 <= cp <= MAX_UNICODE and not (0xD800 <= cp <= 0xDFFF)

def build_map():
    """The reference's BuildMapInternal (the NFKD decompositions mapped
    to their NFKC forms, the unnormalized variants expanded, then
    RemoveRedundantMap), with the two measured overrides: the codepoints
    the reference's older tables do not map, and the deletions."""
    nfkd_decomposed = set()
    norm2orig = {}
    rules = {}   # tuple(key cps) -> tuple(target cps)
    for cp in range(1, MAX_UNICODE + 1):
        if not is_unicode_char(cp):
            continue
        ch = chr(cp)
        nfkc = unicodedata.normalize("NFKC", ch)
        if cp not in REF_UNMAPPED and (len(nfkc) >= 2 or
                                       (len(nfkc) == 1 and ord(nfkc) != cp)):
            rules[(cp,)] = tuple(ord(c) for c in nfkc)
        nfkd = unicodedata.normalize("NFKD", ch)
        if len(nfkd) == 1:

            if cp not in REF_UNMAPPED:
                norm2orig.setdefault(ord(nfkd), set()).add(cp)
        else:
            nfkd_decomposed.add(tuple(ord(c) for c in nfkd))
    for cp in REF_DELETED:
        rules[(cp,)] = ()
    rules.update(_seq_rules(nfkd_decomposed, norm2orig))
    return remove_redundant(rules)

def _seq_rules(nfkd_decomposed, norm2orig):
    out = {}
    for nfkd in nfkd_decomposed:
        nfkc = tuple(ord(c) for c in
                     unicodedata.normalize("NFKC",
                                           "".join(chr(c) for c in nfkd)))
        if nfkc == nfkd:
            continue
        for orig in expand_unnormalized(nfkd, norm2orig):
            if orig != nfkc and orig not in REF_UNMAPPED_SEQ:
                out[orig] = nfkc
    return out

def expand_unnormalized(nfkd, norm2orig):
    results = [(c,) for c in sorted(norm2orig[nfkd[0]])]
    for c in nfkd[1:]:
        new = []
        for r in results:
            for o in sorted(norm2orig[c]):
                new.append(r + (o,))
        results = new
    return results

def normalize(chars_map, src, max_len):
    out = []
    i = 0
    while i < len(src):
        key = tuple(src[i:i + max_len])
        while key and key not in chars_map:
            key = key[:-1]
        if not key:
            out.append(src[i])
            i += 1
        else:
            out.extend(chars_map[key])
            i += len(key)
    return tuple(out)

def remove_redundant(chars_map):
    max_len = max(len(k) for k in chars_map)
    new_map = {k: v for k, v in chars_map.items() if len(k) == 1}
    for ln in range(2, max_len + 1):
        for k, v in chars_map.items():
            if len(k) == ln and v != normalize(new_map, k, ln - 1):
                new_map[k] = v
    for k, v in chars_map.items():
        assert v == normalize(new_map, k, max_len), (k, v)
    return new_map

def canonize(cp, rules):
    """The C runtime's spm_canon: the single-codepoint rule's target
    when that target is exactly one codepoint, else the codepoint."""
    v = rules.get((cp,))
    if v is not None and len(v) == 1:
        return v[0]
    return cp

def split_tables(cm):
    """A = the single-codepoint rules (the canonicalizer plus the plain
    compatibility mappings); C = the canonical multi-codepoint keys.
    Verifies the runtime's equivalence: every multi-cp rule must be the
    variant expansion of exactly one canonical key, so the runtime can
    canonicalize each codepoint and then look up C (this is what makes
    the compact table possible)."""
    A = {k: v for k, v in cm.items() if len(k) == 1}
    C = {}
    for k, v in cm.items():
        if len(k) >= 2 and all(canonize(c, A) == c for c in k):
            C[k] = v

    bad = 0
    for k, v in cm.items():
        if len(k) == 1:
            continue
        ck = tuple(canonize(c, A) for c in k)
        if C.get(ck) != v:
            bad += 1
            if bad <= 5:
                print("  structural: %s -> %s has no canonical key (%s)"
                      % (" ".join("%X" % c for c in k),
                         " ".join("%X" % c for c in v),
                         " ".join("%X" % c for c in ck) if ck in C
                         else "absent"))
    print("tables: A=%d (1-cp)  C=%d (canonical, >=2cp)  structural holes=%d"
          % (len(A), len(C), bad))
    return A, C

def main():
    args = sys.argv[1:]
    cm = build_map()
    by_len = {}
    for k in cm:
        by_len[len(k)] = by_len.get(len(k), 0) + 1
    print("rules: %d  (by key length: %s)" %
          (len(cm), ", ".join("%d->%d" % (k, by_len[k])
                              for k in sorted(by_len))))
    tlen = max(len(v) for v in cm.values())
    print("max target length: %d  deletions: %d"
          % (tlen, sum(1 for v in cm.values() if not v)))
    A, C = split_tables(cm)
    if "--check" in args:
        check_single(cm)
    if "--emit" in args:
        emit(A, C, args[args.index("--emit") + 1])

def emit_pool(f, name, cps, per_line=12):
    f.write("const uint32_t %s[] = {\n" % name)
    for i in range(0, len(cps), per_line):
        f.write("    " + ", ".join("0x%X" % c for c in cps[i:i + per_line])
                + ",\n")
    f.write("    0x0\n};\n\n")

def emit(A, C, src_dir):
    """Writes src/spm_charmap.h (A: cp -> target) and src/spm_seqmap.h
    (C: the canonical key -> target trie, longest match)."""
    keys = sorted(A)
    strs = []
    entries = []
    for k in keys:
        cp = k[0]
        v = A[k]
        entries.append((cp, len(strs), len(v)))
        strs.extend(v)
    with open(src_dir + "/spm_charmap.h", "w") as f:
        f.write("#ifndef TSIMD_SPM_CHARMAP_H\n#define TSIMD_SPM_CHARMAP_H\n")
        f.write("#include <stdint.h>\n")
        f.write("typedef struct { uint32_t cp; uint32_t off; uint8_t len; }"
                " tsimd_spm_map_t;\n")
        f.write("extern const uint32_t tsimd_spm_charmap_str[];\n")
        f.write("extern const tsimd_spm_map_t tsimd_spm_charmap[];\n")
        f.write("extern const int tsimd_spm_charmap_n;\n")
        f.write("extern const uint8_t tsimd_spm_charmap_bits[];\n")
        f.write("extern const uint32_t tsimd_spm_charmap_high[];\n")
        f.write("extern const int tsimd_spm_charmap_high_n;\n\n")
        emit_pool(f, "tsimd_spm_charmap_str", strs)
        f.write("const tsimd_spm_map_t tsimd_spm_charmap[] = {\n")
        for cp, off, ln in entries:
            f.write("    {0x%X, %d, %d},\n" % (cp, off, ln))
        f.write("};\n\n")
        f.write("const int tsimd_spm_charmap_n = %d;\n\n" % len(entries))

        bits = bytearray(8192)
        high = []
        for k in keys:
            cp = k[0]
            if cp < 0x10000:
                bits[cp >> 3] |= 1 << (cp & 7)
            else:
                high.append(cp)
        f.write("const uint8_t tsimd_spm_charmap_bits[] = {\n")
        for i in range(0, len(bits), 16):
            f.write("    " + ", ".join("0x%02X" % b for b in bits[i:i + 16])
                    + ",\n")
        f.write("};\n\n")
        f.write("const uint32_t tsimd_spm_charmap_high[] = {\n")
        for i in range(0, len(high), 12):
            f.write("    " + ", ".join("0x%X" % c for c in high[i:i + 12])
                    + ",\n")
        f.write("    0x0\n};\n\n")
        f.write("const int tsimd_spm_charmap_high_n = %d;\n\n#endif\n"
                % len(high))
    print("wrote %s/spm_charmap.h: %d rules, %d pooled cps"
          % (src_dir, len(entries), len(strs)))

    keys = sorted(C)
    strs = []
    entries = []
    for k in keys:
        v = C[k]
        entries.append((len(strs), len(k), len(v)))
        strs.extend(k)
        strs.extend(v)
    maxkey = max(len(k) for k in keys)
    with open(src_dir + "/spm_seqmap.h", "w") as f:
        f.write("#ifndef TSIMD_SPM_SEQMAP_H\n#define TSIMD_SPM_SEQMAP_H\n")
        f.write("#include <stdint.h>\n")
        f.write("typedef struct { uint32_t off; uint8_t klen; uint8_t tlen; }"
                " tsimd_spm_seq_t;\n")
        f.write("extern const uint32_t tsimd_spm_seq_str[];\n")
        f.write("extern const tsimd_spm_seq_t tsimd_spm_seq[];\n")
        f.write("extern const int tsimd_spm_seq_n;\n")
        f.write("#define TSIMD_SPM_SEQ_MAXKEY %d\n\n" % maxkey)
        emit_pool(f, "tsimd_spm_seq_str", strs)
        f.write("const tsimd_spm_seq_t tsimd_spm_seq[] = {\n")
        for off, klen, tlen in entries:
            f.write("    {%d, %d, %d},\n" % (off, klen, tlen))
        f.write("};\n\n")
        f.write("const int tsimd_spm_seq_n = %d;\n\n#endif\n" % len(entries))
    print("wrote %s/spm_seqmap.h: %d keys, %d pooled cps, max key %d"
          % (src_dir, len(entries), len(strs), maxkey))

def check_single(cm):
    """The generated single-codepoint rules must equal the run-66 table
    (same targets, byte for byte, up to the string layout)."""
    import re
    txt = open("src/spm_charmap.h").read()
    strs = re.search(r"spm_charmap_str\[\] = \{(.*?)\};", txt, re.S)
    body = re.search(r"spm_charmap\[\] = \{(.*?)\};", txt, re.S)
    if strs is None or body is None:
        print("check: no previous table in src/spm_charmap.h (nothing to compare)")
        return
    cps = [int(x, 0) for x in re.findall(r"0x[0-9A-Fa-f]+", strs.group(1))]
    body = re.search(r"spm_charmap\[\] = \{(.*?)\};", txt, re.S)
    old = {}
    for m in re.finditer(r"\{0x([0-9A-Fa-f]+), (\d+), (\d+)\}", body.group(1)):
        cp, off, ln = int(m.group(1), 16), int(m.group(2)), int(m.group(3))
        old[cp] = tuple(cps[off:off + ln])
    new = {k[0]: v for k, v in cm.items() if len(k) == 1}
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    diff = sorted(cp for cp in set(old) & set(new) if old[cp] != new[cp])
    print("single-cp: old=%d new=%d  only-old=%d only-new=%d target-diff=%d"
          % (len(old), len(new), len(only_old), len(only_new), len(diff)))
    for cp in only_old[:10]:
        print("  only-old U+%04X -> %s" % (cp, " ".join("%X" % c for c in old[cp])))
    for cp in only_new[:10]:
        print("  only-new U+%04X -> %s" % (cp, " ".join("%X" % c for c in new[cp])))
    for cp in diff[:10]:
        print("  diff U+%04X old=%s new=%s" % (cp,
              " ".join("%X" % c for c in old[cp]),
              " ".join("%X" % c for c in new[cp])))

if __name__ == "__main__":
    main()
