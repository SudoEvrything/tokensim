#include "tokenizer.h"
#include "batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"
#define NEMO TSIMD_SRC_DIR "/models/reference/nemotron3_nano_tokenizer.json"
#define MIB (1u << 20)
#define NFULL 4u
#define NDOCS 64u

static int fails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static const char *PAT =
    "The quick brown fox jumps over the lazy dog. Caf\xc3\xa9 na\xc3\xafve "
    "\xe2\x80\x94 0123456789 <tag attr=\"v\"> text </tag> a  b   c    \n"
    "unicorn \xf0\x9f\xa6\x84 mixed \xe4\xb8\xad\xe6\x96\x87 right.\n";

static void fill(uint8_t *buf, size_t len) {
    size_t n = strlen(PAT);
    for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)PAT[i % n];
}

static int encode_single(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                         uint32_t **ids, size_t *n_ids) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    *ids = malloc(seq.count ? seq.count * sizeof(uint32_t) : 1);
    if (!*ids) {
        tsimd_token_sequence_free(&seq);
        return -1;
    }
    memcpy(*ids, seq.ids, seq.count * sizeof(uint32_t));
    *n_ids = seq.count;
    tsimd_token_sequence_free(&seq);
    return 0;
}

static void one_model(const char *path, const char *tag) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, path, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "%s: load failed\n", tag);
        fails++;
        return;
    }
    size_t len = NFULL * MIB;
    uint8_t *data = malloc(len);
    check(data != NULL, "input buffer");
    if (!data) {
        tsimd_tokenizer_free(&t);
        return;
    }
    fill(data, len);

    uint32_t *ref = NULL;
    size_t n_ref = 0;
    check(encode_single(&t, data, len, &ref, &n_ref) == 0, "single encode");
    check(n_ref > 0, "single encode is not empty");

    static const int TH[] = {1, 2, 3, 4, 6, 8};
    for (size_t k = 0; k < sizeof TH / sizeof TH[0]; k++) {
        tsimd_input_span_t docs[1];
        docs[0].data = data;
        docs[0].length = len;
        tsimd_token_batch_t b;
        if (tsimd_tokenizer_encode_batch(&t, docs, 1, &b, TH[k]) != 0) {
            fprintf(stderr, "%s: one-doc batch failed at %d threads\n", tag,
                    TH[k]);
            fails++;
            continue;
        }
        char what[128];
        snprintf(what, sizeof what, "%s: one doc, %d threads, id count", tag,
                 TH[k]);
        check(b.id_count == n_ref, what);
        if (b.id_count == n_ref)
            check(memcmp(b.ids, ref, n_ref * sizeof(uint32_t)) == 0,
                  "one doc, ids equal to the single encode");
        tsimd_token_batch_free(&b);
    }

    {
        tsimd_input_span_t docs[2];
        docs[0].data = data;
        docs[0].length = len;
        docs[1].data = data;
        docs[1].length = len;
        tsimd_token_batch_t b;
        if (tsimd_tokenizer_encode_batch(&t, docs, 2, &b, 4) != 0) {
            fprintf(stderr, "%s: two-doc batch failed\n", tag);
            fails++;
        } else {
            check(b.id_count == 2 * n_ref, "two docs, id count");
            check(b.offsets[0] == 0 && b.offsets[1] == n_ref &&
                      b.offsets[2] == 2 * n_ref,
                  "two docs, offsets");
            if (b.id_count == 2 * n_ref)
                check(memcmp(b.ids + b.offsets[1], ref,
                             n_ref * sizeof(uint32_t)) == 0,
                      "two docs, the second half equals the single encode");
            tsimd_token_batch_free(&b);
        }
    }

    {
        size_t dlen = len / NDOCS;
    }
    {
        size_t dlen = len / NDOCS;
        tsimd_input_span_t *docs = calloc(NDOCS, sizeof(*docs));
        uint32_t **per = calloc(NDOCS, sizeof(*per));
        size_t *pern = calloc(NDOCS, sizeof(*pern));
        check(docs && per && pern, "per-doc buffers");
        if (docs && per && pern) {
            for (size_t d = 0; d < NDOCS; d++) {
                docs[d].data = data + d * dlen;
                docs[d].length = dlen;
                if (encode_single(&t, docs[d].data, dlen, &per[d],
                                  &pern[d]) != 0)
                    fails++;
            }
            for (size_t k = 0; k < sizeof TH / sizeof TH[0]; k++) {
                tsimd_token_batch_t b;
                if (tsimd_tokenizer_encode_batch(&t, docs, NDOCS, &b, TH[k]) !=
                    0) {
                    fprintf(stderr, "%s: %u-doc batch failed at %d threads\n",
                            tag, NDOCS, TH[k]);
                    fails++;
                    continue;
                }
                size_t want = 0;
                for (size_t d = 0; d < NDOCS; d++) want += pern[d];
                check(b.id_count == want, "many docs, id count");
                if (b.id_count != want || b.offsets[1] > want)
                    fprintf(stderr,
                            "  th=%d: id_count %zu (want %zu) n_docs %zu "
                            "offsets[0]=%zu offsets[1]=%zu offsets[2]=%zu\n",
                            TH[k], b.id_count, want, b.n_docs, b.offsets[0],
                            b.offsets[1], b.offsets[2]);
                int ok = b.id_count == want && b.offsets[0] == 0;
                for (size_t d = 0; ok && d < NDOCS; d++) {
                    if (b.offsets[d + 1] - b.offsets[d] != pern[d]) ok = 0;
                    else if (memcmp(b.ids + b.offsets[d], per[d],
                                    pern[d] * sizeof(uint32_t)) != 0)
                        ok = 0;
                }
                if (!ok) {
                    for (size_t d = 0; d < NDOCS; d++) {
                        size_t bl = b.offsets[d + 1] - b.offsets[d];
                        if (bl != pern[d]) {
                            fprintf(stderr,
                                    "  doc %zu: batch %zu ids, single %zu\n",
                                    d, bl, pern[d]);
                            continue;
                        }
                        size_t i = 0;
                        while (i < bl &&
                               b.ids[b.offsets[d] + i] == per[d][i])
                            i++;
                        if (i < bl)
                            fprintf(stderr,
                                    "  doc %zu: first diff at %zu of %zu "
                                    "(batch %u, single %u)\n",
                                    d, i, bl, b.ids[b.offsets[d] + i],
                                    per[d][i]);
                    }
                }
                check(ok, "many docs, every slice equals its single encode");
                tsimd_token_batch_free(&b);
            }
            for (size_t d = 0; d < NDOCS; d++) free(per[d]);
        }
        free(docs);
        free(per);
        free(pern);
    }

    free(ref);
    free(data);
    tsimd_tokenizer_free(&t);
}

int main(void) {
    one_model(QWEN, "qwen2.5");
    one_model(NEMO, "nemotron3");
    if (fails) {
        fprintf(stderr, "%d batch scale checks FAILED\n", fails);
        return 1;
    }
    printf("test_batch_scale: one-doc, two-doc and %u-doc batches agree with "
           "the single encode at 1..8 threads\n",
           (unsigned)NDOCS);
    tsimd_nfc_tls_release();
    return 0;
}
