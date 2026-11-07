#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump_spans(const char *label, const uint8_t *d, size_t n,
                       const tsimd_span_array_t *sp) {
    printf("%s n=%zu spans=%u:", label, n, sp->count);
    for (uint32_t i = 0; i < sp->count; i++) {
        uint32_t s = sp->spans[i].start, e = sp->spans[i].end;
        printf(" [");
        for (uint32_t k = s; k < e; k++)
            if (d[k] >= 0x20 && d[k] < 0x7F) printf("%c", d[k]);
            else printf("<%02x>", d[k]);
        printf(")");
    }
    printf("\n");
}

static void run_case(const char *utf8) {
    size_t n = strlen(utf8);
    uint8_t buf[256];
    memcpy(buf, utf8, n);
    memset(buf + n, 'x', sizeof(buf) - n);

    tsimd_span_array_t sp;
    tsimd_span_array_init(&sp);
    if (tsimd_pretokenize(TSIMD_PRETOK_DEEPSEEK, buf, n, &sp) != 0) {
        printf("%s: pretokenize FAILED\n", utf8);
    } else {
        dump_spans("pretokenize", buf, n, &sp);
    }
    tsimd_span_array_free(&sp);

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_DEEPSEEK;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON,
                                   "models/reference/deepseek_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0) {
        printf("load failed\n");
        return;
    }
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(&tok, buf, n, &seq) != 0) {
        printf("encode failed\n");
    } else {
        printf("encode ids:");
        for (size_t i = 0; i < seq.count; i++) printf(" %u", seq.ids[i]);
        printf("\n");
    }
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&tok);
}

int main(void) {
    const char *cases[] = {
        ")\u30fb\"S",    "\u30fb\"S",     "\"Sam",
        "\u30fb\"Sam",   "(\"Sam",        "a\"S",
        "x\")\u30fb\"S", "\"\u30fbS",     "\u30fb\" Sam",
    };
    for (unsigned i = 0;
         i < sizeof(cases) / sizeof(cases[0]); i++)
        run_case(cases[i]);
    return 0;
}
