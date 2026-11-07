#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void show(const char *label, const tsimd_vocab_t *v,
                 tsimd_bpe_workspace_t *ws, const uint8_t *bytes, uint32_t len) {
    ws->out_count = 0;
    if (tsimd_bpe_encode_spm(v, ws, bytes, len) != 0) {
        printf("%s: encode error\n", label);
        return;
    }
    printf("%s (%u bytes) ->", label, len);
    for (size_t i = 0; i < ws->out_count; i++) {
        uint32_t pl;
        const uint8_t *pb = tsimd_vocab_bytes(v, ws->out[i], &pl);
        printf(" [%u]", ws->out[i]);
        for (uint32_t k = 0; k < pl; k++) {
            if (pb[k] >= 0x20 && pb[k] < 0x7f)
                printf("%c", pb[k]);
            else
                printf("<%02x>", pb[k]);
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model\n", argv[0]);
        return 2;
    }
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, argv[1],
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, tok.vocab.max_token_len) != 0) return 1;
    const tsimd_vocab_t *v = &tok.vocab;

    show("word 'SP<html>'", v, &ws,
         (const uint8_t *)"\xe2\x96\x81<html>", 3 + 6);
    show("word 'SP<html>SP<head>'", v, &ws,
         (const uint8_t *)"\xe2\x96\x81<html>\xe2\x96\x81<head>",
         3 + 6 + 3 + 5);
    show("run '.com SP<html>SP<head>'", v, &ws,
         (const uint8_t *)".com\xe2\x96\x81<html>\xe2\x96\x81<head>",
         4 + 3 + 6 + 3 + 5);

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    const uint8_t raw[] = " <html> <head>";
    if (tsimd_tokenizer_encode(&tok, raw, 14, &seq) == 0) {
        printf("sequential ' <html> <head>' ->");
        for (size_t i = 0; i < seq.count; i++) printf(" [%u]", seq.ids[i]);
        printf("\n");
        tsimd_token_sequence_free(&seq);
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&tok);
    return 0;
}
