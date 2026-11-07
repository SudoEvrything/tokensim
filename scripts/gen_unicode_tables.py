#!/usr/bin/env python3
"""Generate src/unicode_tables.c: sorted codepoint ranges mapping
non-ASCII codepoints to pretokenizer character classes (letter/number/
whitespace) plus o200k case-class flags (Lu/Lt/Lm/Lo/M vs Ll/Lm/Lo/M).
ASCII is handled inline in C. Regenerate after a new Unicode standard
is needed."""
import sys
import unicodedata

WS_CODEPOINTS = (
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)

def classify(cp):
    if cp in WS_CODEPOINTS:
        return 3  # SPACE
    cat = unicodedata.category(chr(cp))
    if cat.startswith("L"):
        return 1  # LETTER
    if cat.startswith("N"):
        return 2  # NUMBER
    return 0  # OTHER

def cat_flags(cp):
    r"""The DeepSeek-body category flags (run 59): bit0 = \p{P}|\p{S}
    (the body regex only uses the union), bit1 = \p{M}, bit2 = \p{Cf}."""
    cat = unicodedata.category(chr(cp))
    f = 0
    if cat.startswith("P") or cat.startswith("S"): f |= 1
    if cat.startswith("M"): f |= 2
    if cat == "Cf": f |= 4
    return f

def case_flags(cp):
    """o200k letter-run case classes (matters for the case-split rules):
    1 = Upper (Lu/Lt): U-phase only; ends a lowercase-phase run
    2 = Lower (Ll/Lo): L-phase; Lo (CJK) anchors L+ immediately
    3 = Caseless (Lm): dual-class, U phase + backtrack point
    4 = Mark (Mn/Mc/Me): dual-class, no phase change, backtrack point"""
    cat = unicodedata.category(chr(cp))
    if cat in ("Lu", "Lt"):
        return 1
    if cat in ("Ll", "Lo"):
        return 2
    if cat == "Lm":
        return 3
    if cat.startswith("M"):
        return 4
    return 0

def main(out_path):
    ranges = []  # (start, end_inclusive, cls, cat)
    cur_start, cur_cls, cur_cat = None, None, 0
    for cp in range(0x80, 0x110000):
        c = classify(cp) if not (0xD800 <= cp <= 0xDFFF) else 0
        f = 0 if 0xD800 <= cp <= 0xDFFF else cat_flags(cp)
        if c != cur_cls or (cur_cls is not None and f != cur_cat):
            if cur_cls is not None and (cur_cls != 0 or cur_cat != 0):
                ranges.append((cur_start, cp - 1, cur_cls, cur_cat))
            cur_start, cur_cls, cur_cat = cp, c, f
    if cur_cls not in (None, 0) or (cur_cls == 0 and cur_cat != 0):
        ranges.append((cur_start, 0x10FFFF, cur_cls, cur_cat))

    case_ranges = []
    cur_start, cur_val = None, 0
    for cp in range(0x80, 0x110000):
        v = 0 if 0xD800 <= cp <= 0xDFFF else case_flags(cp)
        if v != cur_val:
            if cur_val:
                case_ranges.append((cur_start, cp - 1, cur_val))
            cur_start, cur_val = cp, v
    if cur_val:
        case_ranges.append((cur_start, 0x10FFFF, cur_val))

    with open(out_path, "w") as f:

        f.write('#include "unicode.h"\n\n')
        f.write("typedef struct { uint32_t lo, hi; uint8_t cls, cat; } "
                "tsimd_urange_t;\n\n")
        f.write("static const tsimd_urange_t k_ranges[] = {\n")
        for lo, hi, cls, cat in ranges:
            f.write("    {0x%X, 0x%X, %d, %d},\n" % (lo, hi, cls, cat))
        f.write("};\n\n")
        f.write("#define K_RANGE_COUNT %du\n" % len(ranges))
        f.write("static const tsimd_urange_t k_case_ranges[] = {\n")
        for lo, hi, val in case_ranges:
            f.write("    {0x%X, 0x%X, %d},\n" % (lo, hi, val))
        f.write("};\n\n")
        f.write("#define K_CASE_RANGE_COUNT %du\n" % len(case_ranges))
        f.write("\nconst uint16_t tsimd_bmp_props[0xFF80] = {\n")
        row = []
        for cp in range(0x80, 0x10000):
            if 0xD800 <= cp <= 0xDFFF:
                c, cat, ca = 0, 0, 0
            else:
                c, cat, ca = classify(cp), cat_flags(cp), case_flags(cp)
            row.append(c | (cat << 3) | (ca << 6))
            if len(row) == 12:
                f.write("    " + ", ".join("0x%03X" % v for v in row) + ",\n")
                row = []
        if row:
            f.write("    " + ", ".join("0x%03X" % v for v in row) + ",\n")
        f.write("};\n\n")
        f.write("""int tsimd_bmp_props_selftest(uint32_t *n_bad) {
    uint32_t bad = 0;
    for (uint32_t cp = 0x80; cp < 0x10000; cp++) {
        uint16_t got = tsimd_bmp_props[cp - 0x80];
        uint32_t cls = 0, cat = 0;
        size_t lo = 0, hi = K_RANGE_COUNT;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (cp < k_ranges[mid].lo) hi = mid;
            else if (cp > k_ranges[mid].hi) lo = mid + 1;
            else { cls = k_ranges[mid].cls; cat = k_ranges[mid].cat; break; }
        }
        uint32_t ca = 0;
        size_t clo = 0, chi = K_CASE_RANGE_COUNT;
        while (clo < chi) {
            size_t mid = clo + (chi - clo) / 2;
            if (cp < k_case_ranges[mid].lo) chi = mid;
            else if (cp > k_case_ranges[mid].hi) clo = mid + 1;
            else { ca = k_case_ranges[mid].cls; break; }
        }
        uint16_t want = (uint16_t)(cls | (cat << 3) | (ca << 6));
        if (got != want) bad++;
    }
    if (n_bad) *n_bad = bad;
    return bad ? -1 : 0;
}
""")
        body = """
tsimd_char_class_t tsimd_class_codepoint(uint32_t cp) {
    if (cp < 0x80) return tsimd_class_ascii((uint8_t)cp);
    if (cp < 0x10000) return (tsimd_char_class_t)(tsimd_bmp_props[cp - 0x80] & 7);
    size_t lo = 0, hi = K_RANGE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < k_ranges[mid].lo) hi = mid;
        else if (cp > k_ranges[mid].hi) lo = mid + 1;
        else return (tsimd_char_class_t)k_ranges[mid].cls;
    }
    return TSIMD_CLASS_OTHER;
}

uint32_t tsimd_class_codepoint_flags(uint32_t cp) {
    if (cp < 0x80) {
        uint32_t c = (uint32_t)tsimd_class_ascii((uint8_t)cp);
        uint8_t b = (uint8_t)cp;
        uint32_t f = 0;
        if ((b >= 0x21 && b <= 0x2F) || (b >= 0x3A && b <= 0x40) ||
            (b >= 0x5B && b <= 0x60) || (b >= 0x7B && b <= 0x7E))
            f |= 1;
        return c | (f << 3);
    }
    if (cp < 0x10000) return (uint32_t)(tsimd_bmp_props[cp - 0x80] & 0x3F);
    size_t lo = 0, hi = K_RANGE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < k_ranges[mid].lo) hi = mid;
        else if (cp > k_ranges[mid].hi) lo = mid + 1;
        else return ((uint32_t)k_ranges[mid].cls) |
                    (((uint32_t)k_ranges[mid].cat) << 3);
    }
    return TSIMD_CLASS_OTHER;
}

uint8_t tsimd_class_case(uint32_t cp) {
    if (cp < 0x80) {
        if (cp >= 'A' && cp <= 'Z') return 1;
        if (cp >= 'a' && cp <= 'z') return 2;
        return 0;
    }
    if (cp < 0x10000) return (uint8_t)((tsimd_bmp_props[cp - 0x80] >> 6) & 7);
    size_t lo = 0, hi = K_CASE_RANGE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < k_case_ranges[mid].lo) hi = mid;
        else if (cp > k_case_ranges[mid].hi) lo = mid + 1;
        else return k_case_ranges[mid].cls;
    }
    return 0;
}
"""
        f.write(body)
    print("wrote %s: %d class ranges, %d case ranges" % (out_path, len(ranges), len(case_ranges)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "src/unicode_tables.c")
