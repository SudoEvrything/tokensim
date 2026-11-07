#include "tokenizer.h"
#include "vocabulary.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;

static void chk(int cond, const char *what, unsigned long a,
                unsigned long b) {
    if (!cond) {
        fprintf(stderr, "FAIL %s: %lu vs %lu\n", what, a, b);
        g_fail = 1;
    }
}

int main(void) {
    static const char *extras[] = {"ab", "ba", "abab", "aab", "abba"};
    size_t len;
    char *buf = tsimd_test_make_vocab(extras, 5, &len);
    if (!buf) return 2;

    tsimd_vocab_t v;
    memset(&v, 0, sizeof v);
    if (tsimd_vocab_load_buffer((const uint8_t *)buf, len, &v) != 0) {
        fprintf(stderr, "FAIL: load\n");
        return 1;
    }

    uint32_t id = 0;
    uint32_t p;

    p = tsimd_vocab_merge_prio_id(&v, 'a', 'b', &id);
    chk(p != UINT32_MAX && id == 256, "ab merge", p, 256);

    p = tsimd_vocab_merge_prio_id(&v, 'b', 'a', &id);
    chk(p != UINT32_MAX && id == 257, "ba merge", p, 257);

    p = tsimd_vocab_merge_prio_id(&v, 256, 256, &id);
    chk(p != UINT32_MAX && id == 258, "abab merge", p, 258);

    p = tsimd_vocab_merge_prio_id(&v, 'a', 256, &id);
    chk(p != UINT32_MAX && id == 259, "aab merge", p, 259);

    p = tsimd_vocab_merge_prio_id(&v, 256, 257, &id);
    chk(p != UINT32_MAX && id == 260, "abba merge", p, 260);

    p = tsimd_vocab_merge_prio_id(&v, 'a', 'a', &id);
    chk(p == UINT32_MAX, "aa is not a merge", p, UINT32_MAX);

    p = tsimd_vocab_merge_prio_id(&v, 257, 'b', &id);
    chk(p == UINT32_MAX, "bab is not a merge", p, UINT32_MAX);

    tsimd_vocab_free(&v);
    free(buf);
    printf("  reconstruct order: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail;
}
