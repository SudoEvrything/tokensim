#include "unicode.h"

#include <stdio.h>

extern const uint16_t tsimd_bmp_props[0xFF80];
int tsimd_bmp_props_selftest(uint32_t *n_bad);

static int g_fail;

static void chk(int cond, const char *what, unsigned long got,
                unsigned long want) {
    if (!cond) {
        fprintf(stderr, "FAIL %s: got %lu want %lu\n", what, got, want);
        g_fail = 1;
    }
}

int main(void) {
    uint32_t bad = 0;
    int rc = tsimd_bmp_props_selftest(&bad);
    chk(rc == 0 && bad == 0, "bmp table vs ranges (all 65408 codepoints)",
        (unsigned long)bad, 0);

    chk((tsimd_bmp_props[0xE9 - 0x80] & 7) == TSIMD_CLASS_LETTER,
        "U+00E9 is a letter", tsimd_bmp_props[0xE9 - 0x80] & 7,
        TSIMD_CLASS_LETTER);
    chk((tsimd_bmp_props[0x2014 - 0x80] & 7) == TSIMD_CLASS_OTHER,
        "U+2014 is other", tsimd_bmp_props[0x2014 - 0x80] & 7,
        TSIMD_CLASS_OTHER);
    chk((tsimd_bmp_props[0x0661 - 0x80] & 7) == TSIMD_CLASS_NUMBER,
        "U+0661 is a number", tsimd_bmp_props[0x0661 - 0x80] & 7,
        TSIMD_CLASS_NUMBER);
    chk((tsimd_bmp_props[0x00A0 - 0x80] & 7) == TSIMD_CLASS_SPACE,
        "U+00A0 is a space", tsimd_bmp_props[0x00A0 - 0x80] & 7,
        TSIMD_CLASS_SPACE);

    chk(tsimd_class_codepoint(0xE9) == TSIMD_CLASS_LETTER,
        "class_codepoint U+00E9",
        (unsigned long)tsimd_class_codepoint(0xE9), TSIMD_CLASS_LETTER);
    chk(tsimd_class_codepoint(0x4E2D) == TSIMD_CLASS_LETTER,
        "class_codepoint U+4E2D",
        (unsigned long)tsimd_class_codepoint(0x4E2D), TSIMD_CLASS_LETTER);
    chk(tsimd_class_case(0x41) == 1, "case U+0041",
        (unsigned long)tsimd_class_case(0x41), 1);
    chk(tsimd_class_case(0xE9) == 2, "case U+00E9",
        (unsigned long)tsimd_class_case(0xE9), 2);
    chk(tsimd_class_case(0x2014) == 0, "case U+2014",
        (unsigned long)tsimd_class_case(0x2014), 0);
    chk(tsimd_class_case(0x10400) == 1, "case U+10400 (astral path)",
        (unsigned long)tsimd_class_case(0x10400), 1);
    chk((tsimd_class_codepoint_flags(0x2014) & 7) == TSIMD_CLASS_OTHER,
        "flags U+2014 class",
        (unsigned long)(tsimd_class_codepoint_flags(0x2014) & 7),
        TSIMD_CLASS_OTHER);
    chk(((tsimd_class_codepoint_flags(0x2014) >> 3) & 7) != 0,
        "flags U+2014 has a category",
        (unsigned long)((tsimd_class_codepoint_flags(0x2014) >> 3) & 7), 1);
    chk((tsimd_class_codepoint_flags(0x4E2D) & 7) == TSIMD_CLASS_LETTER,
        "flags U+4E2D class",
        (unsigned long)(tsimd_class_codepoint_flags(0x4E2D) & 7),
        TSIMD_CLASS_LETTER);

    if (g_fail) {
        fprintf(stderr, "test_bmp_props: FAILED\n");
        return 1;
    }
    printf("test_bmp_props: OK (all BMP codepoints verified)\n");
    return 0;
}
