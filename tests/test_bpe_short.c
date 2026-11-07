#include "bpe.h"
#include "util.h"
#include "vocabulary.h"
#include "tokenizer.h"

#include <stdio.h>

#define TSIMD_NIL UINT32_MAX
#include <stdlib.h>
#include <string.h>

#define NAIVE_MAX 64

static int naive_bpe(const tsimd_vocab_t *v, const uint8_t *bytes,
                     uint32_t len, uint32_t *out_ids, uint32_t *n_out) {
    uint32_t whole = tsimd_vocab_find(v, bytes, len);
    if (whole != UINT32_MAX) {
        out_ids[0] = whole;
        *n_out = 1;
        return 0;
    }
    uint32_t ids[NAIVE_MAX], nxt[NAIVE_MAX], prv[NAIVE_MAX];
    (void)prv;
    for (uint32_t i = 0; i < len; i++) {
        ids[i] = v->byte_ids[bytes[i]];
        prv[i] = (i == 0) ? TSIMD_NIL : i - 1;
        nxt[i] = (i + 1 == len) ? TSIMD_NIL : i + 1;
    }
    uint32_t count = len;
    while (count > 1) {

        uint32_t best = UINT32_MAX, best_i = TSIMD_NIL;
        for (uint32_t i = 0; i < len; i++) {
            uint32_t j = nxt[i];
            if (j >= NAIVE_MAX) continue;
            uint32_t r = tsimd_vocab_merge_priority(v, ids[i], ids[j]);
            if (r < best) {
                best = r;
                best_i = i;
            }
        }
        if (best == UINT32_MAX) break;
        ids[best_i] = tsimd_vocab_id_of_priority(v, best);
        uint32_t dead = nxt[best_i];
        uint32_t after = nxt[dead];
        nxt[best_i] = after;
        nxt[dead] = NAIVE_MAX + 1;
        if (after < NAIVE_MAX) prv[after] = best_i;
        count--;
    }
    uint32_t n = 0;
    for (uint32_t p = 0; p < len; p = nxt[p]) {
        if (p >= NAIVE_MAX) break;
        out_ids[n++] = ids[p];
    }
    *n_out = n;
    return 0;
}

static int check_case(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                      const char *label, const uint8_t *bytes, uint32_t len) {
    uint32_t ref_ids[NAIVE_MAX + 1];
    uint32_t n_ref = 0, n_got = 0;
    if (naive_bpe(v, bytes, len, ref_ids, &n_ref) != 0) return 1;
    ws->out_count = 0;
    if (tsimd_bpe_encode(v, ws, bytes, len) != 0) return 1;
    n_got = ws->out_count;
    if (n_ref != n_got ||
        memcmp(ref_ids, ws->out, (size_t)n_ref * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "FAIL %s (len %u): naive %u ids [", label, len, n_ref);
        for (uint32_t i = 0; i < n_ref; i++) fprintf(stderr, "%u ", ref_ids[i]);
        fprintf(stderr, "] vs short %u ids [", n_got);
        for (uint32_t i = 0; i < n_got; i++)
            fprintf(stderr, "%u ", ws->out[i]);
        fprintf(stderr, "]\n  bytes:");
        for (uint32_t i = 0; i < len; i++) fprintf(stderr, " %02x", bytes[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    return 0;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {

    static const char *EXTRAS[] = {"ab", "abc", "abcd", " x", "abab",
                                   "abababab"};
    tsimd_vocab_t v;
    size_t vlen;
    char *vbuf = tsimd_test_make_vocab(EXTRAS, 6, &vlen);
    if (!vbuf) { fprintf(stderr, "FAIL: make_vocab\n"); return 1; }
    if (tsimd_vocab_load_buffer((const uint8_t *)vbuf, vlen, &v) != 0) {
        fprintf(stderr, "FAIL: vocab load\n");
        free(vbuf);
        return 1;
    }
    free(vbuf);

    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) { fprintf(stderr, "FAIL: workspace\n"); return 1; }

    static const char *cases[] = {
        "a", "ab", "abc", "abcd", "abdc", "abab", "abababab", "aaaa",
        "xyz", " x", " xab", "aaaaaa", "aaaaaaa", "aaaaaaaa",
        "abcdabcd", "abcabcabc", "ababd", "abxb", "aaaaaaaaaaaaaa",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t len = (uint32_t)strlen(cases[i]);
        if (len > NAIVE_MAX) continue;
        if (check_case(&v, &ws, "hand", (const uint8_t *)cases[i], len))
            return 1;
    }
    printf("  hand cases: OK\n");

    static const char pool[] = "abx z";
    uint8_t buf[NAIVE_MAX];
    for (int iter = 0; iter < 20000; iter++) {
        uint32_t n = (uint32_t)(rnd() % (NAIVE_MAX - 1)) + 2;
        for (uint32_t i = 0; i < n; i++)
            buf[i] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];
        if (check_case(&v, &ws, "fuzz", buf, n)) return 1;
    }
    printf("  20000 fuzz strings: OK\n");

    tsimd_bpe_workspace_free(&ws);
    printf("bpe short-merge oracle: OK\n");
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
    tsimd_nfc_tls_release();
    return 0;
}
