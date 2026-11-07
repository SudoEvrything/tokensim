#!/usr/bin/env python3
"""Generate src/nfc_tables.c: the Unicode NFC/NFKC (UAX #15) data tables.

Four table groups, all derived from Python's unicodedata (the same
source as scripts/gen_unicode_tables.py):
  - the FULL canonical decomposition (NFD) of every codepoint that
    changes under NFD (singletons included; <= 4 codepoints) — the NFC
    engine's table;
  - the FULL compatibility decomposition (NFKD) of every codepoint that
    changes under NFKD (singletons and tagged mappings included; <= 18
    codepoints) — the NFKC engine's table. NFKD output is already the
    recursive fixpoint, so the runtime does one lookup per codepoint;
  - the nonzero canonical combining classes (shared by both engines);
  - the canonical composition pairs (a, b) -> c, filtered exactly like
    the standard requires: the pair composes iff NFC(a + b) == c. The
    composition phase is IDENTICAL for NFC and NFKC (NFKC = NFKD ->
    canonical order -> canonical composition), so one table serves both.

Hangul syllables (U+AC00..U+D7A3) are excluded: the runtime composes
and decomposes them algorithmically. The generator asserts the
invariants the chunked normalizer relies on (see runs 67 and 68): every
composition pair's second element is non-ASCII, every decomposition is
within its bound, no compatibility output contains a Hangul syllable,
and the emitted worst-case expansion constants cover every mapping.
"""
import math
import sys
import unicodedata as ud

SBASE, SCOUNT = 0xAC00, 11172

def main(out_path):
    decomp, cdecomp, ccc, comp = [], [], [], []
    for cp in range(0x80, 0x110000):
        if 0xD800 <= cp <= 0xDFFF or SBASE <= cp < SBASE + SCOUNT:
            continue
        ch = chr(cp)
        nfd = ud.normalize("NFD", ch)
        if nfd != ch:
            cps = [ord(c) for c in nfd]
            assert len(cps) <= 4, (hex(cp), nfd)
            decomp.append((cp, cps))
        nfkd = ud.normalize("NFKD", ch)
        if nfkd != ch:
            cps = [ord(c) for c in nfkd]
            assert len(cps) <= 18, (hex(cp), nfkd)
            assert all(c < 0x80 or not SBASE <= c < SBASE + SCOUNT
                       for c in cps), \
                "a compatibility output contains a Hangul syllable"
            cdecomp.append((cp, cps))
        k = ud.combining(ch)
        if k:
            ccc.append((cp, k))

        dec_str = ud.decomposition(ch)
        if dec_str and not dec_str.startswith("<"):
            parts = [int(x, 16) for x in dec_str.split()]
            if len(parts) == 2:
                a, b = parts
                if ud.normalize("NFC", chr(a) + chr(b)) == ch:
                    comp.append((a, b, cp))
    comp.sort()
    assert all(b >= 0x80 for _, b, _ in comp), \
        "a composition pair has an ASCII second element: chunked NFC unsafe"

    worst = 1
    out_worst = 1
    for cp, cps in cdecomp:
        B = len(chr(cp).encode("utf-8"))
        worst = max(worst, math.ceil(len(cps) / B))
        nfkc = ud.normalize("NFKC", chr(cp))
        out_worst = max(out_worst,
                        math.ceil(len(nfkc.encode("utf-8")) / B))
    for cp, cps in cdecomp:
        B = len(chr(cp).encode("utf-8"))
        assert len(cps) <= worst * B, (hex(cp), len(cps), worst, B)
        assert len(ud.normalize("NFKC", chr(cp)).encode("utf-8")) <= \
            out_worst * B, (hex(cp), out_worst, B)

    flat, rows = [], []
    for cp, cps in decomp:
        rows.append((cp, len(flat), len(cps)))
        flat.extend(cps)
    cflat, crows = [], []
    for cp, cps in cdecomp:
        crows.append((cp, len(cflat), len(cps)))
        cflat.extend(cps)

    out = []

    out.append('#include "nfc_internal.h"\n\n')
    out.append("const uint32_t tsimd_nfc_decomp_cps[] = {\n")
    for i in range(0, len(flat), 8):
        out.append("    " + ", ".join("0x%X" % c for c in flat[i:i+8]) + ",\n")
    out.append("};\n\n")
    out.append("const tsimd_nfc_decomp_t tsimd_nfc_decomp[] = {\n")
    for cp, off, n in rows:
        out.append("    {0x%X, %d, %d},\n" % (cp, off, n))
    out.append("};\nconst size_t tsimd_nfc_decomp_n = sizeof(tsimd_nfc_decomp) / sizeof(tsimd_nfc_decomp[0]);\n\n")
    out.append("const tsimd_nfc_ccc_t tsimd_nfc_ccc[] = {\n")
    for cp, k in ccc:
        out.append("    {0x%X, %d},\n" % (cp, k))
    out.append("};\nconst size_t tsimd_nfc_ccc_n = sizeof(tsimd_nfc_ccc) / sizeof(tsimd_nfc_ccc[0]);\n\n")

    blocks = {}
    for cp, k in ccc:
        blocks.setdefault(cp >> 8, [0] * 256)[cp & 0xFF] = k
    blob, idx = [], []
    off = 1
    blob.append(0)   # the offsets are 1-based (0 = no block); the

    for b in range(0x110000 >> 8):
        if b in blocks:
            idx.append(off)
            blob.extend(blocks[b])
            off += 256
        else:
            idx.append(0)
    assert off <= 0xFFFF, "ccc blob exceeds uint16 offsets"
    out.append("const uint16_t tsimd_nfc_ccc_idx[] = {\n")
    for i in range(0, len(idx), 16):
        out.append("    " + ", ".join(str(v) for v in idx[i:i+16]) + ",\n")
    out.append("};\n\n")
    out.append("const uint8_t tsimd_nfc_ccc_blk[] = {\n")
    for i in range(0, len(blob), 16):
        out.append("    " + ", ".join(str(v) for v in blob[i:i+16]) + ",\n")
    out.append("};\n\n")
    out.append("const tsimd_nfc_pair_t tsimd_nfc_comp[] = {\n")
    for a, b, c in comp:
        out.append("    {0x%X, 0x%X, 0x%X},\n" % (a, b, c))
    out.append("};\nconst size_t tsimd_nfc_comp_n = sizeof(tsimd_nfc_comp) / sizeof(tsimd_nfc_comp[0]);\n\n")

    out.append("const uint32_t tsimd_nfkc_decomp_cps[] = {\n")
    for i in range(0, len(cflat), 8):
        out.append("    " + ", ".join("0x%X" % c for c in cflat[i:i+8]) + ",\n")
    out.append("};\n\n")
    out.append("const tsimd_nfc_decomp_t tsimd_nfkc_decomp[] = {\n")
    for cp, off, n in crows:
        out.append("    {0x%X, %d, %d},\n" % (cp, off, n))
    out.append("};\nconst size_t tsimd_nfkc_decomp_n = sizeof(tsimd_nfkc_decomp) / sizeof(tsimd_nfkc_decomp[0]);\n\n")
    out.append("const size_t tsimd_nfkc_worst = %d;\n" % worst)
    out.append("const size_t tsimd_nfkc_out_worst = %d;\n" % out_worst)
    with open(out_path, "w") as f:
        f.write("".join(out))
    print("%s: decomps %d (%d cps), ccc %d, comp pairs %d, "
          "compat decomps %d (%d cps), worst %d, out_worst %d, unicode %s"
          % (out_path, len(rows), len(flat), len(ccc), len(comp),
             len(crows), len(cflat), worst, out_worst, ud.unidata_version))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "src/nfc_tables.c")
