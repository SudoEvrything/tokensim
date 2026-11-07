#include "tokenizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerInitialize(int *argc, char ***argv);

static tsimd_tokenizer_t *tok;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    tok = malloc(sizeof(tsimd_tokenizer_t));
    memset(tok, 0, sizeof(tsimd_tokenizer_t));
    tsimd_pretok_scheme_t sch0 = TSIMD_PRETOK_O200K;
    if (tsimd_tokenizer_load_model(tok, TSIMD_MODEL_TIKTOKEN,
                                   TSIMD_SRC_DIR "/"
                                   "models/o200k_base.tiktoken",
                                   NULL, &sch0, NULL, 0) != 0) {
        free(tok);
        tok = NULL;
    }
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!tok || size < 2 || size > 512) return 0;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(tok, data, size, &seq) != 0) return 0;
    tsimd_byte_sequence_t dec;
    memset(&dec, 0, sizeof dec);
    if (tsimd_tokenizer_decode(tok, seq.ids, seq.count, 0, &dec) != 0) {
        tsimd_token_sequence_free(&seq);
        return 0;
    }
    if (dec.len != size || memcmp(dec.bytes, data, size) != 0) {
        tsimd_byte_sequence_free(&dec);
        tsimd_token_sequence_free(&seq);
        abort();
    }
    tsimd_byte_sequence_free(&dec);
    tsimd_token_sequence_free(&seq);
    return 0;
}
