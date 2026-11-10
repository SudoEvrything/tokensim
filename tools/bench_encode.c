











#include "tokenizer.h"
#include "input.h"

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
    const char *format = NULL, *scheme_name = NULL;
    const char *pos[3] = {NULL, NULL, NULL};
    const char *path1 = NULL, *path2 = NULL, *corpus = NULL;
    int n_pos = 0;
    int use_read = 0;
    int reuse_out = 0;
    int reps = 3;
    int n_threads = 1;
    size_t mib = 0;
    size_t batch_mib = 0;
    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "--format") == 0 && argi + 1 < argc) {
            format = argv[++argi];
        } else if (strcmp(argv[argi], "--scheme") == 0 && argi + 1 < argc) {
            scheme_name = argv[++argi];
        } else if (strcmp(argv[argi], "--reps") == 0 && argi + 1 < argc) {
            reps = atoi(argv[++argi]);
        } else if (strcmp(argv[argi], "--threads") == 0 && argi + 1 < argc) {
            n_threads = atoi(argv[++argi]);
        } else if (strcmp(argv[argi], "--batch") == 0 && argi + 1 < argc) {



            batch_mib = (size_t)atoi(argv[++argi]);
        } else if (strcmp(argv[argi], "--read") == 0) {




            use_read = 1;
        } else if (strcmp(argv[argi], "--reuse") == 0) {
            reuse_out = 1;
        } else if (strcmp(argv[argi], "--mib") == 0 && argi + 1 < argc) {
            mib = (size_t)atoi(argv[++argi]);
        } else if (n_pos < 3) {
            pos[n_pos++] = argv[argi];
        } else {
            fprintf(stderr, "unexpected arg %s\n", argv[argi]);
            return 2;
        }
    }
    if (n_pos < 2) {
        fprintf(stderr,
                "usage: bench_encode [--format F] [--scheme S] [--reps K] "
                "[--mib N] MODEL [MODEL2] CORPUS\n");
        return 2;
    }
    path1 = pos[0];
    if (n_pos == 3) {
        path2 = pos[1];
        corpus = pos[2];
    } else {
        corpus = pos[1];
    }
    tsimd_model_format_t fmt;
    if (!format || strcmp(format, "tiktoken") == 0) {
        fmt = TSIMD_MODEL_TIKTOKEN;
    } else if (strcmp(format, "gpt2") == 0) {
        fmt = TSIMD_MODEL_GPT2;
    } else if (strcmp(format, "tokenizer_json") == 0) {
        fmt = TSIMD_MODEL_TOKENIZER_JSON;
    } else if (strcmp(format, "sentencepiece") == 0) {
        fmt = TSIMD_MODEL_SENTENCEPIECE;
    } else {
        fprintf(stderr, "unknown format: %s\n", format);
        return 2;
    }
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (scheme_name) {
        if (strcmp(scheme_name, "gpt2") == 0) {
            scheme = TSIMD_PRETOK_GPT2;
        } else if (strcmp(scheme_name, "cl100k") == 0) {
            scheme = TSIMD_PRETOK_CL100K;
        } else if (strcmp(scheme_name, "o200k") == 0) {
            scheme = TSIMD_PRETOK_O200K;
        } else if (strcmp(scheme_name, "spm") == 0) {
            scheme = TSIMD_PRETOK_SPM;
        } else if (strcmp(scheme_name, "gem4") == 0) {
            scheme = TSIMD_PRETOK_GEM4;
        } else if (strcmp(scheme_name, "deepseek") == 0) {


            scheme = TSIMD_PRETOK_DEEPSEEK;
        } else if (strcmp(scheme_name, "nemo") == 0) {
            scheme = TSIMD_PRETOK_NEMO;
        } else if (strcmp(scheme_name, "kimi") == 0) {
            scheme = TSIMD_PRETOK_KIMI;
        } else {
            fprintf(stderr, "unknown scheme: %s\n", scheme_name);
            return 2;
        }
    }

    double t0 = now_sec();
    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load_model(&t, fmt, path1, path2, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    double load_sec = now_sec() - t0;

    tsimd_input_t in;
    if (use_read) {
        if (tsimd_input_read_file(corpus, &in) != 0) {
            fprintf(stderr, "read failed: %s\n", corpus);
            return 1;
        }
    } else if (tsimd_input_mmap_file(corpus, &in) != 0) {
        fprintf(stderr, "mmap failed: %s\n", corpus);
        return 1;
    }
    size_t len = in.length;
    if (mib && len > (mib << 20)) len = mib << 20;
    const uint8_t *data = in.data;
    const double mib_total = (double)len / (1024.0 * 1024.0);

    double best = 0.0;
    size_t expect_count = 0;
    tsimd_token_sequence_t reuse_seq;
    memset(&reuse_seq, 0, sizeof reuse_seq);
    uint32_t *reuse_buf = NULL;
    size_t reuse_cap = 0;
    if (reuse_out && n_threads > 1) {
        reuse_cap = tsimd_tokenizer_encode_bound(&t, len);
        reuse_buf = (uint32_t *)malloc(reuse_cap * sizeof(uint32_t));
        if (!reuse_buf) return 1;
    }
    for (int r = 0; r < reps; r++) {
        double s = now_sec();
        size_t count = 0;
        int rc;
        tsimd_token_sequence_t seq;
        tsimd_token_batch_t batch;
        if (batch_mib > 0) {


            size_t dlen = batch_mib << 20;
            size_t n_docs = (len + dlen - 1) / dlen;
            tsimd_input_span_t *docs = malloc(n_docs * sizeof(*docs));
            if (!docs) return 1;
            for (size_t d = 0; d < n_docs; d++) {
                docs[d].data = data + d * dlen;
                docs[d].length =
                    (d + 1 == n_docs) ? len - d * dlen : dlen;
            }
            rc = tsimd_tokenizer_encode_batch(&t, docs, n_docs, &batch,
                                              n_threads);
            free(docs);
            if (rc == 0) count = batch.id_count;
        } else if (reuse_out && n_threads == 1) {
            rc = tsimd_tokenizer_encode_into(&t, data, len, &reuse_seq);
            seq = reuse_seq;
            if (rc == 0) count = seq.count;
        } else if (reuse_out && n_threads > 1) {
            rc = tsimd_tokenizer_encode_parallel_into(&t, data, len,
                                                      n_threads, reuse_buf,
                                                      reuse_cap, &count);
            seq.count = count;
        } else {
            rc = n_threads > 1
                     ? tsimd_tokenizer_encode_parallel(&t, data, len,
                                                       n_threads, &seq)
                     : tsimd_tokenizer_encode(&t, data, len, &seq);
            if (rc == 0) count = seq.count;
        }
        if (rc != 0) {
            fprintf(stderr, "encode failed (rep %d)\n", r);
            return 1;
        }
        double sec = now_sec() - s;
        if (r == 0) {
            expect_count = count;
        } else if (count != expect_count) {
            fprintf(stderr,
                    "warm-pass checksum guard: rep %d produced %zu ids, "
                    "rep 0 produced %zu\n", r, count, expect_count);
            return 1;
        }
        double mibps = mib_total / sec;
        double mtokps = (double)count / sec / 1e6;
        printf("rep %d  %8.3f s  %8.1f MiB/s  %7.2f Mtok/s  (%zu ids)\n",
               r, sec, mibps, mtokps, count);
        if (mibps > best) best = mibps;
        if (batch_mib > 0)
            tsimd_token_batch_free(&batch);
        else if (!reuse_out)
            tsimd_token_sequence_free(&seq);
    }
    tsimd_input_release(&in);
    tsimd_tokenizer_free(&t);
    if (getenv("TSIMD_MASK_STATS")) {
        extern void tsimd_o200k_mask_stats(size_t *, size_t *);
        tsimd_o200k_mask_stats(NULL, NULL);
    }
    if (reuse_out) tsimd_token_sequence_free(&reuse_seq);
    free(reuse_buf);
    printf("BEST %8.1f MiB/s  (load %.2f s, %zu MiB, %d reps)\n",
           best, load_sec, (size_t)mib_total, reps);
    return 0;
}
