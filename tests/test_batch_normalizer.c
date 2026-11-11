#include "batch.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMMA4 TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json"

static int check_doc(tsimd_tokenizer_t *t, const uint8_t *doc, size_t n,
                     const tsimd_token_batch_t *batch, size_t d) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, doc, n, &seq) != 0) return 1;
    size_t a = batch->offsets[d];
    size_t b = batch->offsets[d + 1];
    int bad = (b - a) != seq.count ||
              memcmp(batch->ids + a, seq.ids, seq.count * 4) != 0;
    tsimd_token_sequence_free(&seq);
    return bad;
}

int main(void) {
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, GEMMA4,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "cannot load gemma4\n");
        return 1;
    }
    if (t.scheme != TSIMD_PRETOK_GEM4) {
        fprintf(stderr, "gemma4 scheme wrong as NFC\n");
        return 1;
    }
    const int ndocs = 4;
    size_t dlen = 1u << 20;
    uint8_t *docs[4];
    tsimd_input_span_t spans[4];
    for (int d = 0; d < ndocs; d++) {
        docs[d] = malloc(dlen);
        unsigned h = 0x9E3779B9u ^ (unsigned)(d * 0x85EBCA6Bu);
        for (size_t i = 0; i < dlen; i++) {
            h = h * 0x01000193u + 1;
            docs[d][i] = (uint8_t)((h >> 24) | 0x80u);
            if ((i & 7u) == 7u) docs[d][i] = ' ';
        }
        spans[d].data = docs[d];
        spans[d].length = dlen;
    }
    tsimd_token_batch_t batch;
    memset(&batch, 0, sizeof batch);
    int rc = tsimd_tokenizer_encode_batch(&t, spans, ndocs, &batch, 6);
    if (rc != 0) {
        fprintf(stderr, "encode_batch failed\n");
        return 1;
    }
    int bad = 0;
    for (int d = 0; d < ndocs; d++)
        bad += check_doc(&t, docs[d], dlen, &batch, (size_t)d);
    for (int d = 0; d < ndocs; d++) free(docs[d]);
    tsimd_token_batch_free(&batch);
    tsimd_tokenizer_free(&t);
    if (bad) {
        fprintf(stderr, "%d batch docs differ from the sequential encodes\n",
                bad);
        return 1;
    }
    printf("test_batch_normalizer: NFC-expanding batch equals the "
           "sequential encodes\n");
    return 0;
}
