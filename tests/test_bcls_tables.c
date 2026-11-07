#include "pretokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define B_TAIL 1
#define B_WS 2
#define B_LETTER 4
#define B_DIGIT 8
#define B_OTHER 16
#define B_HIGH 32
#define B_SPACE 64

extern const uint8_t tsimd_cl100k_bcls[256];
extern const uint8_t tsimd_deepseek_bcls[256];
extern const uint8_t tsimd_o200k_bcls[256];
extern const uint8_t tsimd_nemo_bcls[256];

static int g_fail;
static int checks;

static int is_tail_byte(uint8_t b) { return b == '\r' || b == '\n'; }

static int is_tail_byte_slash(uint8_t b) {
    return b == '\r' || b == '\n' || b == '/';
}

static int is_ascii_ws(uint8_t b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == 0x0b ||
           b == 0x0c;
}

static int is_letter_ascii(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
}

static int is_other_ascii(uint8_t b) {
    return !is_letter_ascii(b) && !(b >= '0' && b <= '9') && !is_ascii_ws(b);
}

static uint8_t oracle_byte_class(uint8_t b, int tail_slash) {
    uint8_t v = 0;
    if (tail_slash ? is_tail_byte_slash(b) : is_tail_byte(b)) v |= B_TAIL;
    if (is_ascii_ws(b)) v |= B_WS;
    if (is_letter_ascii(b)) v |= B_LETTER;
    if (b >= '0' && b <= '9') v |= B_DIGIT;
    if (is_other_ascii(b)) v |= B_OTHER;
    if (b >= 0x80) v |= B_HIGH;
    if (b == ' ') v |= B_SPACE;
    return v;
}

static void check_table(const char *name, const uint8_t *tab, int tail_slash) {
    unsigned long bad = 0;
    uint8_t first_bad = 0, first_want = 0, first_got = 0;
    for (int b = 0; b < 256; b++) {
        uint8_t want = oracle_byte_class((uint8_t)b, tail_slash);
        if (tab[b] != want) {
            if (bad == 0) {
                first_bad = (uint8_t)b;
                first_want = want;
                first_got = tab[b];
            }
            bad++;
        }
    }
    checks++;
    if (bad) {
        fprintf(stderr,
                "FAIL %s: %lu byte(s) differ, first 0x%02x want 0x%02x got "
                "0x%02x\n",
                name, bad, first_bad, first_want, first_got);
        g_fail = 1;
    } else {
        printf("  %s: 256 classes match the predicates\n", name);
    }
}

static uint32_t c2_os_oracle(uint8_t b2, int tail_slash) {
    if (b2 >= 0x80) return 0xFFFFFFFFu;
    if (tail_slash ? is_tail_byte_slash(b2) : is_tail_byte(b2)) return 0;
    return (uint32_t)(b2 == ' ' || is_other_ascii(b2));
}

static void check_c2_os(const char *name, const uint8_t *tab, int tail_slash) {
    unsigned long bad = 0;
    uint8_t first_bad = 0;
    for (int b = 0; b < 256; b++) {
        uint8_t bb = tab[b];
        uint32_t got;
        if (b >= 0x80) got = 0xFFFFFFFFu;
        else got = (bb & B_TAIL) ? 0u : (((bb & B_SPACE) || (bb & B_OTHER)) ? 1u : 0u);
        uint32_t want = c2_os_oracle((uint8_t)b, tail_slash);
        if (got != want) {
            if (bad == 0) first_bad = (uint8_t)b;
            bad++;
        }
    }
    checks++;
    if (bad) {
        fprintf(stderr, "FAIL %s c2_os: %lu byte(s) differ, first 0x%02x\n",
                name, bad, first_bad);
        g_fail = 1;
    }
}

static void check_osf_in(const char *name, const uint8_t *tab) {
    unsigned long bad = 0;
    for (int b = 0; b < 256; b++) {
        for (int p = 0; p < 256; p++) {
            uint8_t bb = tab[b], pb = tab[p];
            uint32_t got =
                (uint32_t)((bb & B_OTHER) && !(pb & B_OTHER) && !(pb & B_WS));
            uint32_t want = (uint32_t)(is_other_ascii((uint8_t)b) &&
                                       !is_other_ascii((uint8_t)p) &&
                                       !is_ascii_ws((uint8_t)p));
            if (got != want) bad++;
        }
    }
    checks++;
    if (bad) {
        fprintf(stderr, "FAIL %s osf_in: %lu pair(s) differ\n", name, bad);
        g_fail = 1;
    }
}

int main(void) {
    check_table("cl100k", tsimd_cl100k_bcls, 0);
    check_table("deepseek", tsimd_deepseek_bcls, 0);
    check_table("o200k", tsimd_o200k_bcls, 1);
    check_table("nemo", tsimd_nemo_bcls, 1);
    check_c2_os("cl100k", tsimd_cl100k_bcls, 0);
    check_c2_os("deepseek", tsimd_deepseek_bcls, 0);
    check_c2_os("o200k", tsimd_o200k_bcls, 1);
    check_c2_os("nemo", tsimd_nemo_bcls, 1);
    check_osf_in("cl100k", tsimd_cl100k_bcls);
    check_osf_in("deepseek", tsimd_deepseek_bcls);
    if (g_fail) {
        fprintf(stderr, "test_bcls_tables: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_bcls_tables: OK (%d checks)\n", checks);
    return 0;
}
