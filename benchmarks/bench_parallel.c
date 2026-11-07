









#include "tokenizer.h"
#include "input.h"
#include "pretokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}




static double time_encode(tsimd_tokenizer_t *t, const uint8_t *data,
                          size_t len, int threads, size_t *tokens_out,
                          int warm) {
    tsimd_token_sequence_t seq;
    double t0 = now_s();
    int rc;
    if (threads <= 1) {
        rc = tsimd_tokenizer_encode(t, data, len, &seq);
    } else {
        rc = tsimd_tokenizer_encode_parallel(t, data, len, threads, &seq);
    }
    double dt = now_s() - t0;
    if (rc != 0) return -1.0;
    if (warm) *tokens_out = seq.count;
    tsimd_token_sequence_free(&seq);
    return dt;
}

int main(int argc, char **argv) {
    const char *format = "tiktoken";
    const char *corpus = "/tmp/bench_nat.txt";
    int batch_mode = 0;
    int stages_mode = 0;
    int argi = 1;
    while (argi < argc && strncmp(argv[argi], "--", 2) == 0) {
        if (strcmp(argv[argi], "--format") == 0) {
            format = argv[argi + 1];
            argi += 2;
        } else if (strcmp(argv[argi], "--batch") == 0) {
            batch_mode = 1;
            argi += 1;
        } else if (strcmp(argv[argi], "--stages") == 0) {
            stages_mode = 1;
            argi += 1;
        } else {
            fprintf(stderr, "unknown option %s\n", argv[argi]);
            return 2;
        }
    }
    if (argc - argi < 2) {
        fprintf(stderr,
                "usage: %s [--format F] [--batch] MODEL SCHEME [CORPUS]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[argi++];
    const char *scheme_s = argv[argi++];
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (strcmp(scheme_s, "cl100k") == 0) scheme = TSIMD_PRETOK_CL100K;


    if (strcmp(scheme_s, "o200k") == 0) scheme = TSIMD_PRETOK_O200K;
    if (argc - argi >= 1) corpus = argv[argi++];

    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t det = scheme;
    tsimd_model_format_t fmt = strcmp(format, "gpt2") == 0
                                  ? TSIMD_MODEL_GPT2
                                  : TSIMD_MODEL_TIKTOKEN;
    if (tsimd_tokenizer_load_model(&t, fmt, model, NULL, &det, NULL, 0) != 0) {
        fprintf(stderr, "cannot load model %s\n", model);
        return 1;
    }

    tsimd_input_t in;
    if (tsimd_input_mmap_file(corpus, &in) != 0 &&
        tsimd_input_read_file(corpus, &in) != 0) {
        fprintf(stderr, "cannot read corpus %s\n", corpus);
        return 1;
    }
    double mib = (double)in.length / (1024.0 * 1024.0);

    printf("bench_parallel: %s, %.1f MiB, corpus=%s%s\n", scheme_s, mib,
           corpus, batch_mode ? ", mode=batch" : "");

    if (stages_mode) {


        tsimd_span_array_t spans;
        tsimd_span_array_init(&spans);
        double t0 = now_s();
        if (tsimd_pretokenize(det, in.data, in.length, &spans) != 0)
            return 1;
        double pretok = now_s() - t0;

        tsimd_bpe_workspace_t ws;
        tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len);
        t0 = now_s();
        for (size_t i = 0; i < spans.count; i++) {
            uint32_t s0 = spans.spans[i].start, e0 = spans.spans[i].end;
            if (tsimd_bpe_encode(&t.vocab, &ws, in.data + s0, e0 - s0) != 0)
                return 1;
        }
        double bpe = now_s() - t0;

        tsimd_token_sequence_t seq;
        tsimd_tokenizer_encode(&t, in.data, in.length, &seq);
        size_t tok_cached = seq.count;
        tsimd_token_sequence_free(&seq);
        t0 = now_s();
        tsimd_tokenizer_encode(&t, in.data, in.length, &seq);
        double full_warm = now_s() - t0;
        tsimd_token_sequence_free(&seq);

        printf("  stages (1 thread, sequential)\n");
        printf("    pretokenize (SIMD dispatch): %6.3f s  %7.1f MiB/s  %zu spans\n",
               pretok, mib / pretok, spans.count);
        printf("    BPE from spans, no cache:    %6.3f s  %7.1f MiB/s\n",
               bpe, mib / bpe);
        printf("    full encode, warm cache:     %6.3f s  %7.1f MiB/s  %zu tokens\n",
               full_warm, mib / full_warm, tok_cached);
        printf("    sum of stage lower bounds:   %6.3f s (overlaps with cache)\n",
               pretok + bpe);
        tsimd_span_array_free(&spans);
        tsimd_bpe_workspace_free(&ws);
        tsimd_input_release(&in);
        tsimd_tokenizer_free(&t);
        return 0;
    }

    printf("  threads    cold(s)   warm(s)  warm MiB/s  warm Mtok/s  speedup\n");

    size_t ref_tokens = 0;
    double base = -1;
    uint32_t *ref_ids = NULL;
    size_t ref_id_count = 0;
    for (int thr = 1; thr <= 16; thr = thr < 4 ? thr * 2 : thr * 2) {
        size_t tokens = 0;
        double secs = 0;
        double cold_s = 0;
        if (batch_mode) {

            enum { MAX_DOCS = 4096 };
            static tsimd_input_span_t docs[MAX_DOCS];
            size_t n_docs = 0;
            size_t doc_len = 1 << 16;
            for (size_t off = 0; off < in.length && n_docs < MAX_DOCS;
                 off += doc_len, n_docs++) {
                docs[n_docs].data = in.data + off;
                docs[n_docs].length =
                    off + doc_len <= in.length ? doc_len : in.length - off;
            }


            tsimd_token_batch_t batch;
            double warm = 0;
            for (int rep = 0; rep < 2; rep++) {
                double t0 = now_s();
                if (tsimd_tokenizer_encode_batch(&t, docs, n_docs, &batch,
                                                thr) != 0) {
                    fprintf(stderr, "batch encode failed\n");
                    return 1;
                }
                double dt = now_s() - t0;
                if (rep == 0) cold_s = dt; else warm = dt;
                tokens = batch.id_count;



                if (thr == 1 && rep == 1) {
                    ref_id_count = batch.id_count;
                    free(ref_ids);
                    ref_ids = malloc(ref_id_count * 4 + 1);
                    if (ref_id_count > 0)
                        memcpy(ref_ids, batch.ids, ref_id_count * 4);
                } else if (rep == 1 &&
                           (batch.id_count != ref_id_count ||
                            (ref_id_count > 0 &&
                             memcmp(batch.ids, ref_ids,
                                    ref_id_count * 4) != 0))) {
                    fprintf(stderr,
                            "batch: %d-thread ids differ from 1-thread\n",
                            thr);
                    return 1;
                }
                tsimd_token_batch_free(&batch);
            }
            secs = warm;
        } else {
            size_t tok_cold = 0;
            cold_s = time_encode(&t, in.data, in.length, thr,
                                 &tok_cold, 0);
            secs = time_encode(&t, in.data, in.length, thr, &tokens, 1);
            if (secs < 0 || cold_s < 0) {
                fprintf(stderr, "encode failed at %d threads\n", thr);
                return 1;
            }
        }
        if (thr == 1) {
            base = secs;
            ref_tokens = tokens;
        }
        printf("  %6d  %9.3f  %8.3f  %10.1f  %11.1f  %5.2fx%s\n", thr,
               cold_s, secs, mib / secs, (double)tokens / secs / 1e6,
               base / secs,
               !batch_mode && tokens != ref_tokens ? " TOKEN MISMATCH" : "");
        if (thr >= 16) break;
    }
    free(ref_ids);

    tsimd_input_release(&in);
    tsimd_tokenizer_free(&t);
    return 0;
}
