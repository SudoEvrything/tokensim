#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerInitialize(int *argc, char ***argv);

static tsimd_tokenizer_t *tok;
static tsimd_pretok_scheme_t sch0 = TSIMD_PRETOK_O200K;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    tok = malloc(sizeof(tsimd_tokenizer_t));
    memset(tok, 0, sizeof(tsimd_tokenizer_t));
    if (tsimd_tokenizer_load_model(tok, TSIMD_MODEL_TIKTOKEN,
                                   TSIMD_SRC_DIR "/"
                                   "models/o200k_base.tiktoken",
                                   NULL, &sch0, NULL, 0) != 0) {
        free(tok);
        tok = NULL;
    }
    return 0;
}

static int oracle_encode(const tsimd_vocab_t *v, const uint8_t *bytes,
                         uint32_t len, uint32_t *out, size_t *out_n) {
    uint32_t ids[4096];
    if (len > 200) len = 200;
    for (uint32_t i = 0; i < len; i++) {
        ids[i] = v->byte_ids[bytes[i]];
    }
    uint32_t n = len;
    while (n > 1) {
        uint32_t best = UINT32_MAX;
        uint32_t best_pos = 0, best_id = 0;
        for (uint32_t k = 0; k + 1 < n; k++) {
            uint32_t id = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(v, ids[k], ids[k + 1],
                                                   &id);
            if (p != UINT32_MAX && p < best) {
                best = p;
                best_pos = k;
                best_id = id;
            }
        }
        if (best == UINT32_MAX) break;
        ids[best_pos] = best_id;
        for (uint32_t k = best_pos + 1; k + 1 < n; k++)
            ids[k] = ids[k + 1];
        n--;
    }
    for (uint32_t i = 0; i < n; i++) out[i] = ids[i];
    *out_n = n;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!tok || size < 2 || size > 200) return 0;
    uint32_t mine[512], oracle[1024];
    size_t mine_n = 512, oracle_n = 512;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(tok, data, size, &seq) != 0) return 0;
    if (seq.count > 500) {
        tsimd_token_sequence_free(&seq);
        return 0;
    }
    memcpy(mine, seq.ids, seq.count * 4);
    mine_n = seq.count;
    tsimd_token_sequence_free(&seq);

    tsimd_span_array_t sp;
    tsimd_span_array_init(&sp);
    if (tsimd_pretokenize(TSIMD_PRETOK_O200K, data, size, &sp) != 0) {
        tsimd_span_array_free(&sp);
        return 0;
    }
    size_t total = 0;
    for (size_t i = 0; i < sp.count; i++) {
        uint32_t ids[512];
        size_t ids_n;
        oracle_encode(&tok->vocab, data + sp.spans[i].start,
                      sp.spans[i].end - sp.spans[i].start, ids, &ids_n);
        if (total + ids_n > sizeof oracle / sizeof oracle[0]) break;
        memcpy(oracle + total, ids, ids_n * 4);
        total += ids_n;
    }
    tsimd_span_array_free(&sp);

    if (mine_n != total ||
        memcmp(mine, oracle, mine_n * 4) != 0) {
        abort();
    }
    return 0;
}
