#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long fnv(const uint32_t *ids, size_t n) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void print_pieces(const tsimd_vocab_t *v, const uint32_t *ids,
                         size_t n, size_t from, size_t to) {
    size_t boff = 0;
    for (size_t i = 0; i < n && i < from; i++) {
        uint32_t pl;
        tsimd_vocab_bytes(v, ids[i], &pl);
        boff += pl;
    }
    for (size_t i = from; i < to && i < n; i++) {
        uint32_t pl;
        const uint8_t *pb = tsimd_vocab_bytes(v, ids[i], &pl);
        printf("[%u@%zu]", ids[i], boff);
        for (uint32_t k = 0; k < pl; k++) {
            if (pb[k] >= 0x20 && pb[k] < 0x7f)
                printf("%c", pb[k]);
            else
                printf("<%02x>", pb[k]);
        }
        printf(" ");
        boff += pl;
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model fixture raw-offset window-len threads "
                "[pad-back] [mode]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *fixture = argv[2];
    size_t off = strtoul(argv[3], NULL, 10);
    size_t wlen = strtoul(argv[4], NULL, 10);
    int threads = atoi(argv[5]);
    size_t back = argc > 6 ? strtoul(argv[6], NULL, 10) : 20480;
    int mode = argc > 7 ? atoi(argv[7]) : 0;

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const tsimd_vocab_t *v = &tok.vocab;
    FILE *f = fopen(fixture, "rb");
    if (!f) {
        fprintf(stderr, "open fixture\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)flen);
    if (!data || fread(data, 1, (size_t)flen, f) != (size_t)flen) return 1;
    fclose(f);
    if (off < back) back = off;
    size_t start = off - back;
    if (start + wlen > (size_t)flen) wlen = (size_t)flen - start;

    tsimd_token_sequence_t sa, sb;
    memset(&sa, 0, sizeof sa);
    memset(&sb, 0, sizeof sb);
    if (mode != 2) {
        if (tsimd_tokenizer_encode(&tok, data + start, wlen, &sa) != 0)
            return 1;
    }
    if (mode != 1) {
        if (tsimd_tokenizer_encode_whole_chain_par(&tok, data + start, wlen,
                                                   threads, &sb) != 0)
            return 1;
    }
    if (mode == 1) {
        printf("encode-only n=%zu fnv=%016llx\n", (size_t)sa.count,
               fnv(sa.ids, sa.count));
        tsimd_token_sequence_free(&sa);
        tsimd_tokenizer_free(&tok);
        free(data);
        return 0;
    }
    if (mode == 2) {
        printf("par-only %dt n=%zu fnv=%016llx\n", threads, (size_t)sb.count,
               fnv(sb.ids, sb.count));
        tsimd_token_sequence_free(&sb);
        tsimd_tokenizer_free(&tok);
        free(data);
        return 0;
    }
    int same = sa.count == sb.count &&
               memcmp(sa.ids, sb.ids, sa.count * sizeof(uint32_t)) == 0;
    printf("window start=%zu len=%zu 1t: n=%zu fnv=%016llx  %dt: n=%zu "
           "fnv=%016llx  %s\n",
           start, wlen, (size_t)sa.count, fnv(sa.ids, sa.count), threads,
           (size_t)sb.count, fnv(sb.ids, sb.count), same ? "MATCH" : "DIFF");
    if (!same) {
        size_t k = 0;
        while (k < sa.count && k < sb.count && sa.ids[k] == sb.ids[k]) k++;
        printf("  first divergent id index %zu\n", k);
        size_t from = k > 3 ? k - 3 : 0;
        print_pieces(v, sa.ids, sa.count, from, k + 6);
        print_pieces(v, sb.ids, sb.count, from, k + 6);
    }
    tsimd_token_sequence_free(&sa);
    tsimd_token_sequence_free(&sb);
    tsimd_tokenizer_free(&tok);
    free(data);
    return same ? 0 : 1;
}
