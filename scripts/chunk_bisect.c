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
                         size_t from, size_t to, size_t *byte_off) {
    for (size_t i = from; i < to; i++) {
        uint32_t pl;
        const uint8_t *pb = tsimd_vocab_bytes(v, ids[i], &pl);
        printf("[%zu@%zu]", ids[i], *byte_off);
        for (uint32_t k = 0; k < pl; k++) {
            if (pb[k] >= 0x20 && pb[k] < 0x7f)
                printf("%c", pb[k]);
            else
                printf("<%02x>", pb[k]);
        }
        printf(" ");
        *byte_off += pl;
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s model corpus chunk-mib threads-a threads-b "
                "[dump-path] [max-dump-chunks]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *corpus = argv[2];
    size_t chunk = (size_t)strtoul(argv[3], NULL, 10) << 20;
    int ta = atoi(argv[4]);
    int tb = atoi(argv[5]);
    const char *dump = argc > 6 ? argv[6] : NULL;
    int maxdump = argc > 7 ? atoi(argv[7]) : 1;

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const tsimd_vocab_t *v = &tok.vocab;

    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 1;
    }
    uint8_t *buf = malloc(chunk + (1u << 20));
    if (!buf) return 1;
    size_t tail = 0;
    size_t got;
    size_t cidx = 0;
    int dumped = 0;
    int mismatched = 0;
    while ((got = fread(buf + tail, 1, chunk, f)) > 0) {
        size_t n = tail + got;
        size_t cut = n;
        if (got == chunk) {
            while (cut > 0 && buf[cut - 1] != ' ') cut--;
            if (cut > 0) cut--;
            if (cut == 0) cut = n;
        }
        tsimd_token_sequence_t sa, sb;
        memset(&sa, 0, sizeof sa);
        memset(&sb, 0, sizeof sb);
        int ra = tsimd_tokenizer_encode_parallel(&tok, buf, (uint32_t)cut, ta,
                                                 &sa);
        int rb = tsimd_tokenizer_encode_parallel(&tok, buf, (uint32_t)cut, tb,
                                                 &sb);
        if (ra != 0 || rb != 0) {
            fprintf(stderr, "encode failed chunk %zu\n", cidx);
            return 1;
        }
        int same = sa.count == sb.count &&
                   memcmp(sa.ids, sb.ids, sa.count * sizeof(uint32_t)) == 0;
        printf("chunk %zu bytes=%zu a: n=%zu fnv=%016llx b: n=%zu fnv=%016llx "
               "%s\n",
               cidx, cut, (size_t)sa.count, fnv(sa.ids, sa.count),
               (size_t)sb.count, fnv(sb.ids, sb.count), same ? "MATCH" : "DIFF");
        if (!same) {
            mismatched++;
            if (dump && dumped < maxdump) {
                FILE *df = fopen(dump, "wb");
                if (df) {
                    fwrite(buf, 1, cut, df);
                    fclose(df);
                    dumped++;
                }
                size_t k = 0;
                while (k < sa.count && k < sb.count && sa.ids[k] == sb.ids[k])
                    k++;
                printf("  first divergent id index %zu of %zu/%zu\n", k,
                       (size_t)sa.count, (size_t)sb.count);
                size_t boff = 0;
                size_t from = k > 4 ? k - 4 : 0;
                print_pieces(v, sa.ids, from,
                             k + 8 < sa.count ? k + 8 : sa.count, &boff);
                boff = 0;
                print_pieces(v, sb.ids, from,
                             k + 8 < sb.count ? k + 8 : sb.count, &boff);
                size_t boff_a = 0;
                for (size_t i = 0; i < k && i < sa.count; i++) {
                    uint32_t pl;
                    tsimd_vocab_bytes(v, sa.ids[i], &pl);
                    boff_a += pl;
                }
                size_t boff_b = 0;
                for (size_t i = 0; i < k && i < sb.count; i++) {
                    uint32_t pl;
                    tsimd_vocab_bytes(v, sb.ids[i], &pl);
                    boff_b += pl;
                }
                printf("  byte offset at divergence: a=%zu b=%zu\n", boff_a,
                       boff_b);
            }
        }
        tsimd_token_sequence_free(&sa);
        tsimd_token_sequence_free(&sb);
        memmove(buf, buf + cut, n - cut);
        tail = n - cut;
        cidx++;
    }
    fclose(f);
    free(buf);
    printf("chunks=%zu mismatched=%d\n", cidx, mismatched);
    tsimd_tokenizer_free(&tok);
    return mismatched ? 1 : 0;
}
