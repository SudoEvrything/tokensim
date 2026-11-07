#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MODEL_NFC TSIMD_SRC_DIR "/models/reference/qwen3_tokenizer.json"
#define MODEL_RAW TSIMD_SRC_DIR "/models/reference/qwen3_nonorm.json"

#define BUF_MIB 4u

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint8_t *gen_ascii(size_t len) {
    static const char *words[] = {
        "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
        "he", "was", "for", "on", "are", "as", "with", "his", "they",
        "I", "at", "be", "this", "have", "from", "or", "one", "had",
        "by", "word", "but", "not", "what", "all", "were", "we", "when",
        "your", "can", "said", "there", "use", "an", "each", "which",
    };
    uint8_t *buf = malloc(len);
    if (!buf) return NULL;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    size_t n = 0;
    while (n < len) {
        const char *w = words[tsimd_test_rand(&rng) %
                              (sizeof(words) / sizeof(words[0]))];
        size_t wl = strlen(w);
        if (n + wl + 2 > len) {
            while (n < len) buf[n++] = 'x';
            break;
        }
        if (n) buf[n++] = ' ';
        memcpy(buf + n, w, wl);
        n += wl;
        if (tsimd_test_rand(&rng) % 97 == 0) buf[n++] = '\n';
    }
    return buf;
}

static int encode_ids(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                      uint32_t **ids, size_t *count) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    *ids = seq.ids;
    *count = seq.count;
    return 0;
}

int main(void) {
    tsimd_tokenizer_t tn, tr;
    memset(&tn, 0, sizeof tn);
    memset(&tr, 0, sizeof tr);
    tsimd_pretok_scheme_t sn = TSIMD_PRETOK_GPT2;
    tsimd_pretok_scheme_t sr = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tn, TSIMD_MODEL_TOKENIZER_JSON,
                                   MODEL_NFC, NULL, &sn, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL_NFC);
        return 2;
    }
    if (tsimd_tokenizer_load_model(&tr, TSIMD_MODEL_TOKENIZER_JSON,
                                   MODEL_RAW, NULL, &sr, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL_RAW);
        return 2;
    }
    TEST_ASSERT_EQ(tn.norm, TSIMD_NORM_NFC);
    TEST_ASSERT_EQ(tr.norm, TSIMD_NORM_NONE);

    size_t len = (size_t)BUF_MIB << 20;
    uint8_t *data = gen_ascii(len);
    if (!data) return 2;

    uint32_t *ids_n = NULL, *ids_r = NULL;
    size_t count_n = 0, count_r = 0;
    double s = now_sec();
    if (encode_ids(&tn, data, len, &ids_n, &count_n) != 0) {
        fprintf(stderr, "nfc encode failed\n");
        return 1;
    }
    double t_nfc = now_sec() - s;
    s = now_sec();
    if (encode_ids(&tr, data, len, &ids_r, &count_r) != 0) {
        fprintf(stderr, "raw encode failed\n");
        return 1;
    }
    double t_raw = now_sec() - s;

    int id_fail = 0;
    if (count_n != count_r) {
        fprintf(stderr, "id count: nfc %zu != raw %zu\n", count_n, count_r);
        id_fail = 1;
    } else if (memcmp(ids_n, ids_r, count_n * sizeof(uint32_t)) != 0) {
        for (size_t i = 0; i < count_n; i++)
            if (ids_n[i] != ids_r[i]) {
                fprintf(stderr, "first id diff at %zu: nfc %u != raw %u\n",
                        i, ids_n[i], ids_r[i]);
                break;
            }
        id_fail = 1;
    }

    int speed_fail = 0;
    if (t_nfc > 2.0 * t_raw + 0.05) {
        fprintf(stderr,
                "nfc encode %.3f s vs raw %.3f s for %u MiB "
                "(the seam walk regressed to the one-byte walk)\n",
                t_nfc, t_raw, BUF_MIB);
        speed_fail = 1;
    }
    printf("qwen3 nfc %.3f s (%.1f MiB/s) vs raw %.3f s (%.1f MiB/s), "
           "%zu ids: %s\n",
           t_nfc, (double)BUF_MIB / t_nfc, t_raw, (double)BUF_MIB / t_raw,
           count_n,
           id_fail ? "IDS WRONG" : (speed_fail ? "SLOW" : "OK"));

    free(ids_n);
    free(ids_r);
    free(data);
    tsimd_tokenizer_free(&tn);
    tsimd_tokenizer_free(&tr);
    if (id_fail || speed_fail) return 1;
    return 0;
}
