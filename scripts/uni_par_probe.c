#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model fmt scheme fixture reps [threads]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    tsimd_model_format_t fmt = strcmp(argv[2], "spm") == 0
                                   ? TSIMD_MODEL_SENTENCEPIECE
                                   : strcmp(argv[3], "tiktoken") == 0
                                         ? TSIMD_MODEL_TIKTOKEN
                                         : TSIMD_MODEL_TOKENIZER_JSON;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, fmt, model, NULL, &sch, NULL, 0) !=
        0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    FILE *fh = fopen(argv[4], "rb");
    if (!fh) {
        fprintf(stderr, "open fixture failed\n");
        return 1;
    }
    static uint8_t raw[1 << 20];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = (size_t)atoi(argv[5]);
    int th = argc > 6 ? atoi(argv[6]) : 6;
    size_t min_reps = ((1u << 18) + rn - 1) / rn;
    size_t max_reps = (8u << 20) / rn;
    if (reps < min_reps) reps = min_reps;
    if (reps > max_reps) reps = max_reps;
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    if (!d) return 1;
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(&t, d, total, &seq) != 0) {
        fprintf(stderr, "seq encode failed\n");
        return 1;
    }
    printf("seq th=1 count=%zu\n", seq.count);
    uint32_t *ref = NULL;
    size_t ref_n = seq.count;
    if (ref_n) {
        ref = malloc(ref_n * 4);
        memcpy(ref, seq.ids, ref_n * 4);
    }
    tsimd_token_sequence_free(&seq);

    if (tsimd_tokenizer_encode_parallel(&t, d, total, th, &seq) != 0) {
        fprintf(stderr, "par encode failed\n");
        return 1;
    }
    printf("par th=%d count=%zu\n", th, seq.count);
    if (seq.count != ref_n) {
        printf("DIVERGE count par=%zu seq=%zu (delta=%zd)\n", seq.count,
               ref_n, (ssize_t)seq.count - (ssize_t)ref_n);
    } else if (ref_n && memcmp(seq.ids, ref, ref_n * 4) != 0) {
        size_t i = 0;
        while (i < ref_n && seq.ids[i] == ref[i]) i++;
        printf("DIVERGE id at %zu: par=%u seq=%u\n", i, seq.ids[i], ref[i]);
    } else {
        printf("MATCH\n");
    }
    {
        tsimd_token_sequence_t wseq;
        memset(&wseq, 0, sizeof wseq);
        if (tsimd_tokenizer_encode_whole_chain_par(&t, d, total, th, &wseq) ==
            0) {
            printf("wchain th=%d count=%zu\n", th, wseq.count);
            if (wseq.count != ref_n)
                printf("WCHAIN-DIVERGE count=%zu seq=%zu\n", wseq.count,
                       ref_n);
            else if (ref_n && memcmp(wseq.ids, ref, ref_n * 4) != 0)
                printf("WCHAIN-DIVERGE ids\n");
            else
                printf("WCHAIN-MATCH\n");
            tsimd_token_sequence_free(&wseq);
        } else {
            printf("wchain FAILED\n");
        }
    }

    tsimd_token_sequence_free(&seq);
    free(ref);
    free(d);
    tsimd_tokenizer_free(&t);
    return 0;
}
