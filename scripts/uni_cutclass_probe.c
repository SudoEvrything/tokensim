#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *CJK_UNIT =
    "数字123と漢字のテスト、日本語のテキスト。 héllo wörld ";

static size_t fill_cjk(uint8_t *buf, size_t n) {
    size_t ul = strlen(CJK_UNIT);
    size_t off = 0;
    while (off < n) {
        size_t take = n - off < ul ? n - off : ul;
        memcpy(buf + off, CJK_UNIT, take);
        off += take;
    }
    return n;
}

static size_t fill_filler(uint8_t *buf, size_t n, int kind) {
    size_t off = 0;
    if (kind == 1) {
        memset(buf, 0xFF, n);
        return n;
    }
    const char *seq4 = "\xf0\x9f\x9a\x80";
    const char *seq4b = "\xf4\x8f\xbf\xbf";
    const char *seq3 = "the";
    const char *s = kind == 2 ? seq4 : (kind == 3 ? seq4b : seq3);
    size_t ul = kind >= 2 && kind <= 3 ? 4 : 3;
    while (off < n) {
        size_t take = n - off < ul ? n - off : ul;
        memcpy(buf + off, s, take);
        off += take;
    }
    return n;
}

static unsigned long long id_hash(const uint32_t *ids, size_t n) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    int kind = argc > 1 ? atoi(argv[1]) : 1;
    size_t fill_n = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 300000u;
    const size_t off = 1940000u;
    const size_t tail = 2000000u;
    size_t n = off + fill_n + tail;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf) return 1;
    fill_cjk(buf, off);
    fill_filler(buf + off, fill_n, kind);
    fill_cjk(buf + off + fill_n, tail);

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &tok, TSIMD_MODEL_SENTENCEPIECE,
            TSIMD_SRC_DIR "/models/reference/t5_spiece.model",
            NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    tsimd_token_sequence_t s1, s2;
    memset(&s1, 0, sizeof s1);
    memset(&s2, 0, sizeof s2);
    int rc1 = tsimd_tokenizer_encode(&tok, buf, n, &s1);
    setenv("TSIMD_UNI_WINDOW", "1", 1);
    int rc2 = tsimd_tokenizer_encode_whole_chain_par(&tok, buf, n, 6, &s2);
    printf("filler=%d raw=%zu seq rc=%d count=%zu hash=%llx\n", kind, n, rc1,
           s1.count, id_hash(s1.ids, s1.count));
    printf("filler=%d raw=%zu win rc=%d count=%zu hash=%llx\n", kind, n, rc2,
           s2.count, id_hash(s2.ids, s2.count));
    if (rc1 == 0 && rc2 == 0) {
        long long delta = (long long)s2.count - (long long)s1.count;
        size_t m = s1.count < s2.count ? s1.count : s2.count;
        size_t d = m;
        for (size_t i = 0; i < m; i++)
            if (s1.ids[i] != s2.ids[i]) {
                d = i;
                break;
            }
        size_t k = 0;
        while (d + k < s1.count && d + 1 + k < s2.count &&
               s1.ids[d + k] == s2.ids[d + 1 + k])
            k++;
        size_t sh0 = 0;
        while (d + sh0 < s1.count && d + sh0 < s2.count &&
               s1.ids[d + sh0] == s2.ids[d + sh0])
            sh0++;
        printf("delta=%lld first_diff=%zu shift0=%zu shift+1=%zu "
               "(seq tail %zu, win tail %zu)\n",
               delta, d, sh0, k, s1.count - d, s2.count - d - 1);
        if (delta == 0 && s1.count == s2.count &&
            id_hash(s1.ids, s1.count) == id_hash(s2.ids, s2.count))
            printf("VERDICT: no split (anchors found in the filler)\n");
        else if (delta > 0 && k == s1.count - d && k == s2.count - d - 1)
            printf("VERDICT: clean split +%lld token(s), tail aligned "
                   "shift+1 (carry exact, no downstream flips)\n",
                   delta);
        else
            printf("VERDICT: split with downstream divergence "
                   "(shift+1 %zu < tail %zu)\n",
                   k, s1.count - d);
    }
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s2);
    tsimd_tokenizer_free(&tok);
    free(buf);
    return 0;
}
