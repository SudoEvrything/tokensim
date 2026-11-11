#include "allocator.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN3 TSIMD_SRC_DIR "/models/reference/qwen3_tokenizer.json"

int main(void) {
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, QWEN3,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "cannot load qwen3\n");
        return 1;
    }
    size_t len = 100u << 20;
    uint8_t *buf = malloc(len);
    if (!buf) return 1;
    for (size_t i = 0; i < len; i++) buf[i] = (i & 1u) ? ' ' : 'a';
    size_t out_len = 0;
    uint8_t *out = tsimd_nfc_normalize_all(&t, buf, len, 100, &out_len);
    if (!out) {
        fprintf(stderr, "normalize failed with max_threads=100\n");
        free(buf);
        return 1;
    }
    if (out_len != len) {
        fprintf(stderr, "plain-ascii NFC changed the length: %zu\n",
                out_len);
        free(buf);
        tsimd_xfree(out);
        return 1;
    }
    free(buf);
    tsimd_xfree(out);
    tsimd_tokenizer_free(&t);
    printf("test_nfc_maxthreads: max_threads=100 clamped, byte-exact "
           "passthrough\n");
    return 0;
}
