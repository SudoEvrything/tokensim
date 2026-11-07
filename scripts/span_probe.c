#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump_hex_spans(const char *label, const uint8_t *d, size_t n,
                           const tsimd_span_array_t *sp) {
    printf("%s spans=%zu:", label, sp->count);
    for (size_t i = 0; i < sp->count; i++) {
        uint32_t s = sp->spans[i].start, e = sp->spans[i].end;
        printf(" [");
        for (uint32_t k = s; k < e; k++) printf("%02x", d[k]);
        printf(")");
    }
    printf("\n");
}

static int scheme_from_name(const char *name, tsimd_pretok_scheme_t *out) {
    if (strcmp(name, "gpt2") == 0) *out = TSIMD_PRETOK_GPT2;
    else if (strcmp(name, "gpt2d") == 0) *out = TSIMD_PRETOK_GPT2D;
    else if (strcmp(name, "cl100k") == 0) *out = TSIMD_PRETOK_CL100K;
    else if (strcmp(name, "o200k") == 0) *out = TSIMD_PRETOK_O200K;
    else if (strcmp(name, "qwen25") == 0) *out = TSIMD_PRETOK_QWEN25;
    else if (strcmp(name, "deepseek") == 0) *out = TSIMD_PRETOK_DEEPSEEK;
    else if (strcmp(name, "nemo") == 0) *out = TSIMD_PRETOK_NEMO;
    else if (strcmp(name, "gem4") == 0) *out = TSIMD_PRETOK_GEM4;
    else return -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s scheme gpt2d hexbytes...\n"
                "       %s --model path scheme hexbytes...\n",
                argv[0], argv[0]);
        return 2;
    }
    int mi = 1;
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    int have_tok = 0;
    if (strcmp(argv[1], "--model") == 0) {
        tsimd_pretok_scheme_t det = TSIMD_PRETOK_GPT2;
        if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON,
                                       argv[2], NULL, &det, NULL, 0) != 0) {
            fprintf(stderr, "load failed\n");
            return 1;
        }
        have_tok = 1;
        mi = 3;
        printf("detected scheme=%d\n", (int)det);
        tsimd_pretok_scheme_t want;
        if (scheme_from_name(argv[mi], &want) != 0) return 2;
        if (want != det)
            printf("NOTE argv scheme %s != detected %d\n", argv[mi],
                   (int)det);
    }
    tsimd_pretok_scheme_t sch;
    if (scheme_from_name(argv[mi], &sch) != 0) return 2;
    mi++;
    uint8_t buf[4096];
    size_t n = 0;
    for (int a = mi; a < argc; a++) {
        const char *h = argv[a];
        for (size_t k = 0; h[k] && h[k + 1] && n < sizeof buf; k += 2) {
            unsigned v;
            if (sscanf(h + k, "%2x", &v) != 1) break;
            buf[n++] = (uint8_t)v;
        }
    }
    tsimd_span_array_t sp;
    tsimd_span_array_init(&sp);
    if (tsimd_pretokenize_scalar_ref(sch, buf, n, &sp) != 0)
        printf("scalar_ref FAILED\n");
    else
        dump_hex_spans("scalar_ref", buf, n, &sp);
    tsimd_span_array_free(&sp);
    tsimd_span_array_init(&sp);
    if (tsimd_pretokenize(sch, buf, n, &sp) != 0)
        printf("pretokenize FAILED\n");
    else
        dump_hex_spans("pretokenize", buf, n, &sp);
    tsimd_span_array_free(&sp);
    {
        int mrc = 0;
        tsimd_span_array_init(&sp);
        switch (sch) {
            case TSIMD_PRETOK_GPT2:
                mrc = tsimd_pretokenize_gpt2_masks(buf, n, &sp);
                break;
            case TSIMD_PRETOK_CL100K:
                mrc = tsimd_pretokenize_cl100k_masks(buf, n, &sp);
                break;
            case TSIMD_PRETOK_GPT2D:
                mrc = tsimd_pretokenize_gpt2d_masks(buf, n, &sp);
                break;
            case TSIMD_PRETOK_DEEPSEEK:
                mrc = tsimd_pretokenize_deepseek_masks(buf, n, &sp);
                break;
            case TSIMD_PRETOK_O200K:
                mrc = tsimd_pretokenize_o200k_masks(buf, n, &sp);
                break;
            case TSIMD_PRETOK_NEMO:
                mrc = tsimd_pretokenize_nemo_masks(buf, n, &sp);
                break;
            default:
                mrc = -2;
        }
        if (mrc == -2)
            printf("masks: (no driver for scheme)\n");
        else if (mrc != 0)
            printf("masks FAILED\n");
        else
            dump_hex_spans("masks", buf, n, &sp);
        tsimd_span_array_free(&sp);
    }
    if (have_tok) {
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode(&tok, buf, n, &seq) == 0) {
            printf("encode ids:");
            for (size_t i = 0; i < seq.count; i++) printf(" %u", seq.ids[i]);
            printf("\n");
        }
        tsimd_token_sequence_free(&seq);
        tsimd_tokenizer_free(&tok);
    }
    return 0;
}
