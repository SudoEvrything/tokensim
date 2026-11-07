#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "bpe.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model fixture reps njobs\n", argv[0]);
        return 2;
    }
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_SPM;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, argv[1],
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    FILE *fh = fopen(argv[2], "rb");
    if (!fh) return 1;
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = (size_t)atoi(argv[3]);
    int njobs = atoi(argv[4]);
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    uint8_t *whole = malloc(total * 3 + 64);
    size_t whole_n = 0;
    size_t w = tsimd_spm_normalize(d, total, whole, total * 3 + 32,
                                   t.spm_dummy_prefix, t.spm_remove_extra);
    if (w == (size_t)-1) {
        printf("whole normalize FAILED\n");
        return 1;
    }
    whole_n = w;
    printf("raw=%zu whole_norm=%zu (x%.3f)\n", total, whole_n,
           (double)whole_n / (double)total);

    size_t slice_norm_total = 0;
    size_t seam = total / (size_t)njobs;
    for (int j = 0; j < njobs; j++) {
        size_t lo = (size_t)j * seam;
        size_t hi = (j == njobs - 1) ? total : lo + seam;
        uint8_t *sout = malloc((hi - lo) * 3 + 64);
        size_t sn = tsimd_spm_normalize(d + lo, hi - lo, sout,
                                        (hi - lo) * 3 + 32,
                                        j == 0 ? t.spm_dummy_prefix : 0,
                                        t.spm_remove_extra);
        if (sn == (size_t)-1) {
            printf("job %d normalize FAILED\n", j);
            return 1;
        }
        slice_norm_total += sn;
        if (j == 0)
            printf("job0 norm=%zu (raw %zu)\n", sn, hi - lo);
        free(sout);
    }
    printf("sum_slice_norm=%zu (x%.3f) delta_vs_whole=%zd\n",
           slice_norm_total, (double)slice_norm_total / (double)total,
           (ssize_t)slice_norm_total - (ssize_t)whole_n);
    free(d);
    free(whole);
    tsimd_tokenizer_free(&t);
    return 0;
}
