#include "tokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO TSIMD_SRC_DIR

int main(void) {
    setenv("TSIMD_PCACHE_STATS", "1", 1);
    setenv("TSIMD_PUBLISH", "1", 1);
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                   REPO "/models/reference/gemma4_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL shared publish: load\n");
        return 1;
    }
    static uint8_t buf[1 << 16];
    size_t len = 0;
    static const char *words[] = {"hello", "world", "the", "token",
                                  "vector", "cache", "stream", "merge"};
    for (int i = 0; i < 6000; i++) {
        const char *w = words[i % 8];
        buf[len++] = 0xE2;
        buf[len++] = 0x96;
        buf[len++] = 0x81;
        for (const char *p = w; *p; p++) buf[len++] = (uint8_t)*p;
        buf[len++] = (uint8_t)('a' + (i % 6));
    }

    size_t h0 = 0, m0 = 0, i0 = 0;
    tsimd_tokenizer_cache_stats(&t, &h0, &m0, &i0);
    size_t sc0 = t.cache.count;

    tsimd_pretok_cache_t priv;
    if (tsimd_pretok_cache_init(&priv, 1u << 12) != 0) return 1;
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0) return 1;
    size_t wt = tsimd_spm_word_target(&t.vocab, 1u << 14);
    if (tsimd_bpe_encode_spm_blocked_cached(&t.vocab, &priv, &t.cache, &ws,
                                            buf, (uint32_t)len, wt) != 0) {
        fprintf(stderr, "FAIL shared publish: rc\n");
        return 1;
    }

    if (t.cache.count <= sc0) {
        fprintf(stderr, "FAIL shared publish: shared count %zu -> %zu "
                        "(nothing published)\n",
                sc0, t.cache.count);
        return 1;
    }
    uint32_t out[4];
    uint32_t n = 0;
    uint8_t probe[8] = {0xE2, 0x96, 0x81, 'h', 'e', 'l', 'l', 'o'};
    uint64_t k0, k1, h;
    tsimd_pcache_key_hash(probe, 8, &k0, &k1, &h);
    if (!tsimd_pcache_lookup_keyed(&t.cache, k0, k1, h, out, &n)) {
        fprintf(stderr, "FAIL shared publish: probe of published unit "
                        "missed (count %zu -> %zu)\n",
                sc0, t.cache.count);
        return 1;
    }
    printf("  shared publish: count %zu -> %zu, probe hits with %u ids\n",
           sc0, t.cache.count, n);

    tsimd_pretok_cache_free(&priv);
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
    printf("shared publish: OK\n");
    return 0;
}
