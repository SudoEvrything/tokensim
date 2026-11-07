#include "model_internal.h"
#include "byte_unicode.h"
#include "unicode.h"
#include "allocator.h"

#include <stdlib.h>
#include <string.h>

int tsimd_decode_remapped(const char *s, size_t n, uint8_t *out, size_t cap,
                         uint32_t *out_len) {
    size_t i = 0, o = 0;
    while (i < n) {
        uint32_t cp;
        size_t adv = tsimd_utf8_decode((const uint8_t *)s + i, n - i, &cp);
        if (adv == 0) return -1;
        uint8_t byte;
        if (!tsimd_cp_to_byte(cp, &byte)) return -1;
        if (o >= cap) return -1;
        out[o++] = byte;
        i += adv;
    }
    *out_len = (uint32_t)o;
    return 0;
}

uint32_t tsimd_vocab_merge_target(const tsimd_vocab_t *v, uint32_t left,
                                 uint32_t right) {
    uint32_t llen, rlen;
    const uint8_t *lb = tsimd_vocab_bytes(v, left, &llen);
    (void)tsimd_vocab_bytes(v, right, &rlen);
    if ((uint64_t)llen + rlen > v->max_token_len) return UINT32_MAX;
    uint8_t *concat = tsimd_xmalloc((size_t)llen + rlen);
    memcpy(concat, lb, llen);
    memcpy(concat + llen, v->bytes + v->offsets[right], rlen);
    uint32_t m = tsimd_vocab_find(v, concat, llen + rlen);
    free(concat);
    return m;
}
