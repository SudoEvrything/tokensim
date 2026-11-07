#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe.h"
#include "tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model fixture reps\n", argv[0]);
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
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    size_t cap = total * 3 + 64;
    uint8_t *nfc = malloc(cap);
    size_t nfc_n = 0;
    {
        uint8_t *n = tsimd_nfc_normalize_all(&t, d, total, 1, &nfc_n);
        if (n) {
            memcpy(nfc, n, nfc_n);
        } else {
            memcpy(nfc, d, total);
            nfc_n = total;
        }
    }
    uint8_t *norm = malloc(cap);
    size_t nn = tsimd_spm_normalize(nfc, nfc_n, norm, cap, t.spm_dummy_prefix,
                                    t.spm_remove_extra);
    if (nn == (size_t)-1) {
        printf("norm failed\n");
        return 1;
    }
    printf("norm=%s nfc_n=%zu norm_len=%zu (raw %zu)\n",
           t.norm == TSIMD_NORM_NFC ? "NFC"
           : t.norm == TSIMD_NORM_NFKC ? "NFKC"
                                       : "NONE",
           nfc_n, nn, total);

    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0) return 1;

    if (tsimd_unigram_encode(&t.vocab, &ws, norm, (uint32_t)nn) != 0) {
        printf("seq dp failed\n");
        return 1;
    }
    printf("seq_dp count=%zu\n", ws.out_count);

    if (tsimd_unigram_encode_par(&t.vocab, &ws, norm, (uint32_t)nn, 6) != 0) {
        printf("par dp failed\n");
        return 1;
    }
    printf("par_dp count=%zu\n", ws.out_count);
    free(d);
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
    return 0;
}
