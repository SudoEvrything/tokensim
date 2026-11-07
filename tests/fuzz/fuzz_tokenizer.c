#include "pretokenizer.h"
#include "tokenizer.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *EXTRAS[] = {
    "ab", "abc", "abcd", " x", "th", "the", " t", " the",
    "er", "her", "in", "ing", "abab", "abababab",
};
#define FUZZ_VOCAB_SIZE (256 + 14)

static int vocab_ready = 0;
static tsimd_tokenizer_t g_tok;

static int ensure_vocab(void) {
    if (vocab_ready) return 0;
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 14, &len);
    if (!buf) return -1;
    int rc = tsimd_vocab_load_buffer((const uint8_t *)buf, len, &g_tok.vocab);
    free(buf);
    if (rc != 0) return -1;
    g_tok.scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_pretok_cache_init(&g_tok.cache, 4096) != 0) return -1;
    vocab_ready = 1;
    return 0;
}

static int spans_equal(const tsimd_span_array_t *a,
                       const tsimd_span_array_t *b) {
    if (a->count != b->count) return 0;
    return a->count == 0 ||
           memcmp(a->spans, b->spans, a->count * sizeof(tsimd_span_t)) == 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 1) return 0;
    if (ensure_vocab() != 0) abort();

    tsimd_pretok_scheme_t scheme = (data[0] & 1) ? TSIMD_PRETOK_CL100K
                                                 : TSIMD_PRETOK_GPT2;
    const uint8_t *input = data + 1;
    size_t len = size - 1;
    if (len > 65536) len = 65536;

    {
        tsimd_span_array_t ref, got;
        tsimd_span_array_init(&ref);
        tsimd_span_array_init(&got);
        if (tsimd_pretokenize_scalar_ref(scheme, input, len, &ref) != 0)
            abort();
        if (tsimd_pretokenize(scheme, input, len, &got) != 0) abort();
        if (!spans_equal(&ref, &got)) {
            fprintf(stderr, "pretokenize mismatch len=%zu\n", len);
            tsimd_span_array_free(&ref);
            tsimd_span_array_free(&got);
            abort();
        }
        tsimd_span_array_free(&ref);
        tsimd_span_array_free(&got);
    }

    {
        g_tok.scheme = scheme;
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&g_tok, input, len, &seq) != 0) abort();
        tsimd_byte_sequence_t dec;
        if (tsimd_tokenizer_decode(&g_tok, seq.ids, seq.count, 0, &dec) != 0)
            abort();
        if (dec.len != len || (len && memcmp(dec.bytes, input, len) != 0)) {
            fprintf(stderr, "round-trip mismatch len=%zu\n", len);
            abort();
        }
        tsimd_byte_sequence_free(&dec);

        if (len > 4) {
            long p = tsimd_pretok_safe_split(scheme, input, len, len / 2, 0);
            if (p > 0 && (size_t)p < len) {
                tsimd_token_sequence_t a, b;
                if (tsimd_tokenizer_encode(&g_tok, input, (size_t)p, &a) != 0)
                    abort();
                if (tsimd_tokenizer_encode(&g_tok, input + (size_t)p,
                                           len - (size_t)p, &b) != 0)
                    abort();
                int ok = a.count + b.count == seq.count;
                for (size_t i = 0; ok && i < a.count; i++)
                    ok = a.ids[i] == seq.ids[i];
                for (size_t i = 0; ok && i < b.count; i++)
                    ok = b.ids[i] == seq.ids[a.count + i];
                if (!ok) {
                    fprintf(stderr, "split invariance failure p=%ld len=%zu\n",
                            p, len);
                    abort();
                }
                tsimd_token_sequence_free(&a);
                tsimd_token_sequence_free(&b);
            }
        }
        tsimd_token_sequence_free(&seq);
    }
    return 0;
}

#ifndef TSIMD_LIBFUZZER_DRIVER

int main(int argc, char **argv) {
    long iterations = argc > 1 ? atol(argv[1]) : 2000;
    uint64_t rng = 0x243F6A8885A308D3ull;
    uint8_t buf[70000];
    for (long it = 0; it < iterations; it++) {
        size_t len = (size_t)(tsimd_test_rand(&rng) % 3000);
        for (size_t i = 0; i < len; i++) {
            uint8_t c;
            switch (rng % 5) {
                case 0: c = (uint8_t)" \t\n\r abcdefghello"[rng % 17]; break;
                case 1: c = (uint8_t)(rng % 128); break;
                case 2: c = (uint8_t)(0x80 + rng % 128); break;
                case 3: c = (uint8_t)"'\"\0/\r\n-"[rng % 8]; break;
                default: c = (uint8_t)(rng % 256); break;
            }
            buf[i] = c;
        }
        buf[len] = 0;
        LLVMFuzzerTestOneInput(buf, len + 1);
    }
    printf("fuzz_tokenizer: %ld iterations OK\n", iterations);
    return 0;
}
#endif
