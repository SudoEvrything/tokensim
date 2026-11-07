#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t fill_cjk(uint8_t *buf, size_t n) {
    const char *unit = "数字123と漢字のテスト、日本語のテキスト。 héllo wörld ";
    size_t ul = strlen(unit);
    size_t off = 0;
    while (off < n) {
        size_t take = n - off < ul ? n - off : ul;
        memcpy(buf + off, unit, take);
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

static void dump_piece(const tsimd_tokenizer_t *t, uint32_t id) {
    const tsimd_vocab_t *v = &t->vocab;
    if (id >= v->vocab_size) {
        printf("<%u>", id);
        return;
    }
    uint32_t len = v->offsets[id + 1] - v->offsets[id];
    const uint8_t *p = v->bytes + v->offsets[id];
    printf("[");
    for (uint32_t i = 0; i < len && i < 24; i++) {
        if (p[i] >= 0x20 && p[i] < 0x7F)
            printf("%c", p[i]);
        else
            printf("\\x%02x", p[i]);
    }
    printf("|%u]", len);
}

int main(int argc, char **argv) {
    size_t n = 4698000u;
    if (argc > 1) n = (size_t)strtoul(argv[1], NULL, 10);
    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf) return 1;
    if (argc > 2) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) {
            fprintf(stderr, "open failed\n");
            free(buf);
            return 1;
        }
        n = fread(buf, 1, n, f);
        fclose(f);
    } else {
        fill_cjk(buf, n);
    }
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
    printf("seq rc=%d count=%zu hash=%llx\n", rc1, s1.count,
           id_hash(s1.ids, s1.count));
    printf("win rc=%d count=%zu hash=%llx\n", rc2, s2.count,
           id_hash(s2.ids, s2.count));
    if (rc1 == 0 && rc2 == 0) {
        size_t m = s1.count < s2.count ? s1.count : s2.count;
        size_t d = m;
        for (size_t i = 0; i < m; i++)
            if (s1.ids[i] != s2.ids[i]) {
                d = i;
                break;
            }
        printf("first_diff=%zu (of %zu common)\n", d, m);
        size_t lo = d > 8 ? d - 8 : 0;
        size_t hi1 = d + 12 < s1.count ? d + 12 : s1.count;
        size_t hi2 = d + 12 < s2.count ? d + 12 : s2.count;
        printf("seq:");
        for (size_t i = lo; i < hi1; i++) {
            printf(" %u", s1.ids[i]);
            dump_piece(&tok, s1.ids[i]);
        }
        printf("\nwin:");
        for (size_t i = lo; i < hi2; i++) {
            printf(" %u", s2.ids[i]);
            dump_piece(&tok, s2.ids[i]);
        }
        printf("\n");
        size_t k = 0;
        while (d + k < s1.count && d + 1 + k < s2.count &&
               s1.ids[d + k] == s2.ids[d + 1 + k])
            k++;
        printf("shift+1 alignment match=%zu (seq tail %zu, win tail %zu)\n", k,
               s1.count - d, s2.count - d - 1);
        k = 0;
        while (d + 1 + k < s1.count && d + k < s2.count &&
               s1.ids[d + 1 + k] == s2.ids[d + k])
            k++;
        printf("shift-1 alignment match=%zu (seq tail %zu, win tail %zu)\n", k,
               s1.count - d - 1, s2.count - d);
    }
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s2);
    tsimd_tokenizer_free(&tok);
    free(buf);
    return 0;
}
