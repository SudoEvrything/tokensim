#include "tokenizer.h"
#include "input.h"
#include "pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    setenv("TSIMD_PCACHE_STATS", "1", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: our_encode_doc MODEL_JSON CORPUS THREADS [MB]\n");
        return 2;
    }
    const char *model = argv[1];
    const char *corpus = argv[2];
    int threads = atoi(argv[3]);
    size_t mb = argc > 4 ? (size_t)atoi(argv[4]) : 500;
    const char *env_mb = getenv("ENCODE_MB");
    if (env_mb && env_mb[0]) mb = (size_t)atoi(env_mb);
    if (threads < 1) threads = 1;

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    {
        const char *es = getenv("ENCODE_SCHEME");
        if (es && es[0]) {
            if (!strcmp(es, "cl100k")) sch = TSIMD_PRETOK_CL100K;
            else if (!strcmp(es, "o200k")) sch = TSIMD_PRETOK_O200K;
            else if (!strcmp(es, "spm")) sch = TSIMD_PRETOK_SPM;
            else if (!strcmp(es, "gem4")) sch = TSIMD_PRETOK_GEM4;
            else if (!strcmp(es, "nemo")) sch = TSIMD_PRETOK_NEMO;
            else if (!strcmp(es, "qwen25")) sch = TSIMD_PRETOK_QWEN25;
            else if (!strcmp(es, "deepseek")) sch = TSIMD_PRETOK_DEEPSEEK;
            else if (!strcmp(es, "kimi")) sch = TSIMD_PRETOK_KIMI;
            else if (!strcmp(es, "gpt2")) sch = TSIMD_PRETOK_GPT2;
        }
    }
    tsimd_model_format_t mt = TSIMD_MODEL_TOKENIZER_JSON;
    size_t mlen = strlen(model);
    if (mlen > 9 && strcmp(model + mlen - 9, ".tiktoken") == 0)
        mt = TSIMD_MODEL_TIKTOKEN;
    else if (mlen > 6 && strcmp(model + mlen - 6, ".model") == 0)
        mt = TSIMD_MODEL_SENTENCEPIECE;
    else if (strstr(model, "vocab.bpe"))
        mt = TSIMD_MODEL_GPT2;
    if (sch == TSIMD_PRETOK_KIMI && mt == TSIMD_MODEL_SENTENCEPIECE)
        mt = TSIMD_MODEL_TIKTOKEN;
    if (tsimd_tokenizer_load_model(&t, mt, model, NULL,
                                   &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", model);
        return 1;
    }

    tsimd_input_t in;
    if (tsimd_input_mmap_file(corpus, &in) != 0) {
        fprintf(stderr, "cannot map %s\n", corpus);
        return 1;
    }
    size_t len = mb << 20;
    if (len > in.length) len = in.length;

    int rounds = 1;
    const char *er = getenv("ENCODE_ROUNDS");
    if (er && er[0]) rounds = atoi(er);
    if (rounds < 1) rounds = 1;

    double size_mb = (double)len / 1e6;
    fprintf(stderr, "1 document, %d threads, scheme %d\n\n", threads,
            (int)sch);
    size_t last_h = 0, last_m = 0, last_i = 0, last_lh = 0, last_lm = 0, last_li = 0;
    tsimd_token_sequence_t reuse_seq;
    memset(&reuse_seq, 0, sizeof reuse_seq);
    const char *einto = getenv("ENCODE_INTO");
    int use_into = (einto && einto[0] == '1');
    for (int r = 0; r < rounds; r++) {
        tsimd_token_sequence_t seq;
        double t0 = now_sec();
        int rc;
        if (use_into && threads == 1) {
            rc = tsimd_tokenizer_encode_into(&t, in.data, len, &reuse_seq);
            seq = reuse_seq;
        } else if (threads == 1) {
            rc = tsimd_tokenizer_encode(&t, in.data, len, &seq);
        } else if (use_into && (getenv("TSIMD_LADDER_INTO") ||
                   tsimd_tokenizer_encode_bound(&t, len) <= (1u << 28))) {
            size_t cap = tsimd_tokenizer_encode_bound(&t, len);
            static uint32_t *pbuf = NULL;
            if (!pbuf) pbuf = malloc(cap * sizeof(uint32_t));
            rc = tsimd_tokenizer_encode_parallel_into(&t, in.data, len,
                                                     threads, pbuf, cap,
                                                     &seq.count);
            seq.ids = pbuf;
        } else {
            rc = tsimd_tokenizer_encode_parallel(&t, in.data, len, threads,
                                                &seq);
        }
        if (rc != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        double el = now_sec() - t0;
        fprintf(stderr, "round %d: %zu tokens in %.3fs — %.0f MB/s\n", r,
                seq.count, el, size_mb / el);
        printf("RESULT %.1f\n", size_mb / el);
        {
            size_t h = 0, m = 0, ins = 0, lh = 0, lm = 0, li = 0;
            tsimd_tokenizer_cache_stats(&t, &h, &m, &ins);
            tsimd_plong_stats(&lh, &lm, &li);
            fprintf(stderr, "cache: hit=%zu miss=%zu ins=%zu | plong: hit=%zu miss=%zu ins=%zu (bits=%u lcount=%zu)\n",
                    h - last_h, m - last_m, ins - last_i, lh - last_lh, lm - last_lm, li - last_li,
                    tsimd_pcache_lbits_of(&t.cache), tsimd_pcache_lcount_of(&t.cache));
            last_h = h; last_m = m; last_i = ins; last_lh = lh; last_lm = lm; last_li = li;
        }
        {
        size_t mp = 0, sp = 0;
        tsimd_o200k_mask_stats(&mp, &sp);
        if (mp || sp) fprintf(stderr, "o200k pieces=%zu scalar_walked=%zu\n", mp, sp);
        {
            size_t cp = 0, cs = 0;
            tsimd_cl100k_mask_stats(&cp, &cs);
            if (cp || cs) fprintf(stderr, "cl100k pieces=%zu scalar_walked=%zu\n", cp, cs);
        }
        {
            size_t dp = 0, ds2 = 0;
            tsimd_deepseek_mask_stats(&dp, &ds2);
            if (dp || ds2) fprintf(stderr, "deepseek mask_pieces=%zu scalar_walked=%zu\n", dp, ds2);
        }
    }
    if (!use_into || seq.ids == NULL) tsimd_token_sequence_free(&seq);
    }
    if (use_into) tsimd_token_sequence_free(&reuse_seq);
    tsimd_input_release(&in);
    tsimd_tokenizer_free(&t);
    return 0;
}
