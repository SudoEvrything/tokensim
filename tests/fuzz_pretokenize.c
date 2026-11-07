#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int LLVMFuzzerInitialize(int *argc, char ***argv);

static tsimd_pretok_scheme_t pick_scheme(const uint8_t *data,
                                         size_t size) {
    static const tsimd_pretok_scheme_t schemes[] = {
        TSIMD_PRETOK_GPT2,  TSIMD_PRETOK_CL100K, TSIMD_PRETOK_QWEN25,
        TSIMD_PRETOK_GPT2D, TSIMD_PRETOK_DEEPSEEK, TSIMD_PRETOK_O200K,
        TSIMD_PRETOK_NEMO,
    };
    if (size < 2) return TSIMD_PRETOK_GPT2;
    return schemes[data[0] % (sizeof schemes / sizeof schemes[0])];
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2) return 0;
    tsimd_pretok_scheme_t sch = pick_scheme(data, size);
    const uint8_t *text = data + 1;
    size_t len = size - 1;

    tsimd_span_array_t a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    if (tsimd_pretokenize(sch, text, len, &a) != 0) {
        tsimd_span_array_free(&a);
        return 0;
    }
    if (tsimd_pretokenize_scalar_ref(sch, text, len, &b) != 0) {
        tsimd_span_array_free(&a);
        tsimd_span_array_free(&b);
        return 0;
    }
    if (a.count != b.count) {
        tsimd_span_array_free(&a);
        tsimd_span_array_free(&b);
        abort();
    }
    for (size_t i = 0; i < a.count; i++) {
        if (a.spans[i].start != b.spans[i].start ||
            a.spans[i].end != b.spans[i].end) {
            tsimd_span_array_free(&a);
            tsimd_span_array_free(&b);
            abort();
        }
    }
    tsimd_span_array_free(&a);
    tsimd_span_array_free(&b);
    return 0;
}
