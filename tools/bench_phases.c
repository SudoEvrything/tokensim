

















#include "tokenizer.h"
#include "pretokenizer.h"
#include "bpe.h"
#include "unicode.h"
#include "input.h"


#include "../src/model_internal.h"

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
    int reps = 3;
    size_t mib = 8;
    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "--format") == 0 && argi + 1 < argc) {
            format = argv[++argi];
        } else if (strcmp(argv[argi], "--scheme") == 0 && argi + 1 < argc) {
            scheme_name = argv[++argi];
        } else if (strcmp(argv[argi], "--reps") == 0 && argi + 1 < argc) {
            reps = atoi(argv[++argi]);
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
                "usage: bench_phases [--format F] [--scheme S] [--reps K] "
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
        } else if (strcmp(scheme_name, "nemo") == 0) {
            scheme = TSIMD_PRETOK_NEMO;
        } else if (strcmp(scheme_name, "qwen25") == 0) {
            scheme = TSIMD_PRETOK_QWEN25;
        } else {
            fprintf(stderr, "unknown scheme: %s\n", scheme_name);
            return 2;
        }
    }

    double t0 = now_sec();
    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load_model(&t, fmt, path1, path2, &scheme, NULL, 0) !=
        0) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    double load_sec = now_sec() - t0;

    tsimd_input_t in;
    if (tsimd_input_mmap_file(corpus, &in) != 0) {
        fprintf(stderr, "mmap failed: %s\n", corpus);
        return 1;
    }
    size_t len = in.length;
    if (mib && len > (mib << 20)) len = mib << 20;
    const uint8_t *data = in.data;
    const double mib_total = (double)len / (1024.0 * 1024.0);

    printf("model load: %.1f s   slice: %.2f MiB\n", load_sec, mib_total);


    double full_best = 0.0;
    size_t expect_ids = 0;
    for (int r = 0; r < reps; r++) {
        double s = now_sec();
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, data, len, &seq) != 0) {
            fprintf(stderr, "encode failed (rep %d)\n", r);
            return 1;
        }
        double sec = now_sec() - s;
        if (r == 0) expect_ids = seq.count;
        else if (seq.count != expect_ids) {
            fprintf(stderr, "guard: rep %d ids %zu != rep 0 %zu\n", r,
                    seq.count, expect_ids);
            return 1;
        }
        double mibps = mib_total / sec;
        printf("full      %8.3f s  %8.1f MiB/s  (%zu ids)\n", sec, mibps,
               seq.count);
        if (mibps > full_best) full_best = mibps;
        tsimd_token_sequence_free(&seq);
    }
    if (full_best <= 0.0) return 1;
    const double full_sec = mib_total / full_best;


    int kind_spm = (t.scheme == TSIMD_PRETOK_SPM && t.spm_kind) ||
                   fmt == TSIMD_MODEL_SENTENCEPIECE;
    int kind_gem4 = t.scheme == TSIMD_PRETOK_GEM4;
    int kind_nfc = t.norm == TSIMD_NORM_NFC || t.norm == TSIMD_NORM_NFKC;
    uint8_t *normbuf = NULL;
    size_t nn = 0;
    if (kind_spm || kind_gem4 || kind_nfc) {
        size_t cap = len * 4 + 64;
        normbuf = malloc(cap);
        if (!normbuf) return 1;

        if (kind_nfc) {
            nn = t.norm == TSIMD_NORM_NFKC
                     ? tsimd_nfkc_normalize(data, len, normbuf, cap)
                     : tsimd_nfc_normalize(data, len, normbuf, cap);
        } else if (kind_gem4) {
            nn = tsimd_metaspace_ex(data, len, normbuf, cap, 0);
        } else {
            nn = tsimd_spm_normalize(data, len, normbuf, cap,
                                     t.spm_dummy_prefix, t.spm_remove_extra);
        }
        if (nn == (size_t)-1) {
            fprintf(stderr, "normalize failed\n");
            return 1;
        }
        double nbest = 0.0;
        for (int r = 0; r < reps; r++) {
            double s = now_sec();
            size_t n2;
            if (kind_nfc) {
                n2 = t.norm == TSIMD_NORM_NFKC
                         ? tsimd_nfkc_normalize(data, len, normbuf, cap)
                         : tsimd_nfc_normalize(data, len, normbuf, cap);
            } else if (kind_gem4) {
                n2 = tsimd_metaspace_ex(data, len, normbuf, cap, 0);
            } else {
                n2 = tsimd_spm_normalize(data, len, normbuf, cap,
                                         t.spm_dummy_prefix,
                                         t.spm_remove_extra);
            }
            if (n2 != nn) {
                fprintf(stderr, "guard: norm rep %d len %zu != %zu\n", r, n2,
                        nn);
                return 1;
            }
            double mibps = mib_total / (now_sec() - s);
            if (mibps > nbest) nbest = mibps;
        }
        printf("norm      %8.3f s  %8.1f MiB/s  %5.1f%% of full\n",
               mib_total / nbest, nbest, 100.0 * (mib_total / nbest) / full_sec);


        if (kind_nfc) {
            double s = now_sec();
            int verdict = 0;
            for (int r = 0; r < reps * 4; r++)
                verdict = tsimd_nfc_may_change(data, len);
            double gate = (now_sec() - s) / (reps * 4);
            printf("nfcmay    %8.3f s  %8.1f MiB/s  %5.1f%% of full "
                   "(verdict=%d: 1 = the real path normalizes)\n",
                   gate, mib_total / gate,
                   100.0 * gate / full_sec, verdict);
        }



        if (kind_nfc) {
            double s = now_sec();
            size_t pos = 0;
            size_t chunks = 0;
            size_t maxlen = 0;
            for (size_t i = 0; i < t.added_count; i++)
                if (t.added[i].length > maxlen) maxlen = t.added[i].length;
            while (pos < len) {
                size_t end;
                size_t want = pos + (((size_t)1) << 20);
                if (want >= len) {
                    end = len;
                } else {

                    size_t from = want;
                    end = len;
                    while (from + 1 < len) {
                        long p = tsimd_pretok_safe_split(
                            t.scheme, data, len, from, 0);
                        if (p < 0) break;
                        size_t seam = (size_t)p;
                        int inside = 0;
                        size_t lo = seam > maxlen ? seam - maxlen : 0;
                        for (size_t q = lo; q < seam; q++) {
                            size_t hit = tsimd_tokenizer_added_len_at(
                                &t, data, len, q, NULL);
                            if (hit > 0 && q + hit > seam) {
                                inside = 1;
                                seam = q + hit;
                                break;
                            }
                        }
                        if (!inside && seam > from) {
                            end = seam;
                            break;
                        }
                        from = seam > from ? seam : from + 1;
                    }
                }
                chunks++;
                tsimd_nfc_may_change(data + pos, end - pos);
                pos = end;
            }
            double sec = now_sec() - s;
            printf("nfcloop   %8.3f s  %8.1f MiB/s  %5.1f%% of full  "
                   "(%zu chunks: the seam walk + the gate)\n",
                   sec, mib_total / sec, 100.0 * sec / full_sec, chunks);
        }
    }


    if (kind_spm || kind_gem4) {
        tsimd_bpe_workspace_t ws;
        if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0)
            return 1;
        double dbest = 0.0;
        size_t expect = 0;
        for (int r = 0; r < reps; r++) {
            ws.out_count = 0;

            double s = now_sec();
            int rc = kind_spm
                         ? tsimd_unigram_encode(&t.vocab, &ws, normbuf,
                                                (uint32_t)nn)
                         : tsimd_bpe_encode_spm(&t.vocab, &ws, normbuf,
                                                (uint32_t)nn);
            if (rc != 0) {
                fprintf(stderr, "dp failed\n");
                return 1;
            }
            double sec = now_sec() - s;
            if (r == 0) expect = ws.out_count;
            else if (ws.out_count != expect) {
                fprintf(stderr, "guard: dp rep %d out %zu != %zu\n", r,
                        ws.out_count, expect);
                return 1;
            }
            double mibps = mib_total / sec;
            printf("dp        %8.3f s  %8.1f MiB/s  %5.1f%% of full\n", sec,
                   mibps, 100.0 * (mib_total / mibps) / full_sec);
            if (mibps > dbest) dbest = mibps;
        }
        (void)dbest;
        tsimd_bpe_workspace_free(&ws);
        free(normbuf);
    } else {
        free(normbuf);

        tsimd_span_array_t spans;
        tsimd_span_array_init(&spans);
        const uint8_t *pdata = data;
        size_t plen = len;
        if (kind_nfc) {

            size_t cap = len * 4 + 64;
            uint8_t *nb = malloc(cap);
            if (!nb) return 1;
            nn = tsimd_nfc_normalize(data, len, nb, cap);
            if (nn == (size_t)-1) return 1;
            pdata = nb;
            plen = nn;
        }
        for (int pass = 0; pass < 2; pass++) {
            double pbest = 0.0;
            size_t expect = 0;
            for (int r = 0; r < reps; r++) {
                spans.count = 0;

                double s = now_sec();
                int rc = pass == 0
                             ? tsimd_pretokenize(t.scheme, pdata, plen,
                                                 &spans)
                             : tsimd_pretokenize_scalar_ref(t.scheme, pdata,
                                                            plen, &spans);
                if (rc != 0) {
                    fprintf(stderr, "pretok failed (pass %d)\n", pass);
                    return 1;
                }
                double sec = now_sec() - s;
                if (r == 0) expect = spans.count;
                else if (spans.count != expect) {
                    fprintf(stderr, "guard: pretok pass %d rep %d spans "
                                    "%zu != %zu\n", pass, r, spans.count,
                            expect);
                    return 1;
                }
                double mibps = mib_total / sec;
                if (mibps > pbest) pbest = mibps;
            }
            printf("%s %8.3f s  %8.1f MiB/s  %5.1f%% of full  (%zu spans)\n",
                   pass == 0 ? "pretok  " : "scalarref",
                   mib_total / pbest, pbest,
                   100.0 * (mib_total / pbest) / full_sec, expect);
        }
        tsimd_span_array_free(&spans);
        if (kind_nfc) free((void *)pdata);
    }

    tsimd_tokenizer_free(&t);
    tsimd_input_release(&in);
    return 0;
}
