#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/reference/qwen3_tokenizer.json"

static uint8_t *gen_nfc_text(size_t len) {

    static const char *parts[] = {
        "caf\xc3\xa9", "e\xcc\x81", "e\xcc\x82\xcc\x81", "hello",
        "na\xc3\xafve", "a\xcc\x80", "world", "test",
        "r\xc3\xa9sum\xc3\xa9", "x\xcc\xa3\xcc\x81",
    };
    uint8_t *buf = malloc(len);
    if (!buf) return NULL;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    size_t n = 0;
    while (n < len) {
        const char *w = parts[rng % (sizeof(parts) / sizeof(parts[0]))];
        size_t wl = strlen(w);
        if (n + wl + 2 > len) {
            while (n < len) buf[n++] = 'x';
            break;
        }
        if (n) buf[n++] = ' ';
        memcpy(buf + n, w, wl);
        n += wl;
        if (rng % 53 == 0 && n < len) buf[n++] = '\n';
    }
    return buf;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, MODEL,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL);
        return 2;
    }
    TEST_ASSERT_EQ(t.norm, TSIMD_NORM_NFC);

    size_t len = (size_t)1 << 20;
    uint8_t *data = gen_nfc_text(len);
    if (!data) return 2;

    size_t cut = len / 2;
    while (cut > 0 && data[cut] != ' ') cut--;
    TEST_ASSERT(cut > 0);

    tsimd_input_span_t docs[2] = {
        {data, cut},
        {data + cut, len - cut},
    };

    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&t, data, len, &seq) != 0) {
        fprintf(stderr, "sequential encode failed\n");
        return 1;
    }

    tsimd_token_batch_t batch;
    if (tsimd_tokenizer_encode_batch(&t, docs, 2, &batch, 4) != 0) {
        fprintf(stderr, "batch encode failed\n");
        return 1;
    }

    int fails = 0;
    if (batch.id_count != seq.count) {
        fprintf(stderr, "id count: batch %zu != sequential %zu\n",
                batch.id_count, seq.count);
        fails++;
    } else if (seq.count &&
               memcmp(batch.ids, seq.ids,
                      seq.count * sizeof(uint32_t)) != 0) {
        for (size_t i = 0; i < seq.count; i++)
            if (batch.ids[i] != seq.ids[i]) {
                fprintf(stderr, "first id diff at %zu: batch %u != seq %u\n",
                        i, batch.ids[i], seq.ids[i]);
                break;
            }
        fails++;
    }
    printf("encode_batch NFC guard: batch %zu ids vs sequential %zu: %s\n",
           batch.id_count, seq.count, fails ? "DIFFS" : "OK");

    tsimd_token_batch_free(&batch);
    tsimd_token_sequence_free(&seq);
    free(data);
    tsimd_tokenizer_free(&t);
    return fails ? 1 : 0;
}
