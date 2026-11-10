#include <tokensim/tokenizer.h>
#include <tokensim/batch.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}


static uint64_t fold(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static uint64_t fold_u64(uint64_t h, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
    return fold(h, b, 8);
}

static uint64_t fold_ids(uint64_t h, const uint32_t *ids, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t b[4];
        b[0] = (uint8_t)ids[i];
        b[1] = (uint8_t)(ids[i] >> 8);
        b[2] = (uint8_t)(ids[i] >> 16);
        b[3] = (uint8_t)(ids[i] >> 24);
        h = fold(h, b, 4);
    }
    return h;
}

static int scheme_of(const char *s, tsimd_pretok_scheme_t *out) {
    if (strcmp(s, "cl100k") == 0) *out = TSIMD_PRETOK_CL100K;
    else if (strcmp(s, "o200k") == 0) *out = TSIMD_PRETOK_O200K;
    else if (strcmp(s, "gpt2") == 0) *out = TSIMD_PRETOK_GPT2;
    else return -1;
    return 0;
}

static int format_of(const char *s, tsimd_model_format_t *out) {
    if (strcmp(s, "tiktoken") == 0) *out = TSIMD_MODEL_TIKTOKEN;
    else if (strcmp(s, "json") == 0) *out = TSIMD_MODEL_TOKENIZER_JSON;
    else if (strcmp(s, "sentencepiece") == 0)
        *out = TSIMD_MODEL_SENTENCEPIECE;
    else if (strcmp(s, "gpt2") == 0) *out = TSIMD_MODEL_GPT2;
    else return -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s MODEL FORMAT SCHEME CORPUS [DOC_BYTES] "
                "[PROMPT_CALLS] [THREADS]\n",
                argv[0]);
        return 2;
    }
    size_t doc_bytes =
        argc > 5 ? (size_t)strtoull(argv[5], NULL, 10) : (size_t)(16u << 20);
    (void)0;
    long prompt_calls = argc > 6 ? strtol(argv[6], NULL, 10) : 20000;
    int threads = argc > 7 ? (int)strtol(argv[7], NULL, 10) : 0;
    if (doc_bytes == 0) doc_bytes = (size_t)(16u << 20);
    if (prompt_calls < 0) prompt_calls = 0;

    tsimd_model_format_t fmt;
    tsimd_pretok_scheme_t sch;
    if (format_of(argv[2], &fmt) != 0 || scheme_of(argv[3], &sch) != 0) {
        fprintf(stderr, "unknown format or scheme\n");
        return 2;
    }
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, fmt, argv[1], NULL, &sch, NULL, 0) !=
        0) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    printf("threads(auto): %d\n", tsimd_default_threads());

    int fd = open(argv[4], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s\n", argv[4]);
        return 1;
    }
    long fsz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    uint8_t *buf = mmap(NULL, (size_t)fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }
    madvise(buf, (size_t)fsz, MADV_SEQUENTIAL);
    size_t ebound = tsimd_tokenizer_encode_bound(&t, doc_bytes);
    uint32_t *ebuf = malloc(ebound * 4);
    if (!ebuf) return 1;

    size_t dbound = doc_bytes + 64;
    uint8_t *dbuf = malloc(dbound);
    if (!dbuf) return 1;
    memset(dbuf, 0, dbound);
    uint64_t h = 0xcbf29ce484222325ULL;
    unsigned long long total = 0, checked = 0, bad = 0;
    size_t docs = 0;
    double enc_t = 0, dec_t = 0;
    double t0 = now();
    for (size_t off = 0; off < (size_t)fsz; off += doc_bytes) {
        size_t n = (size_t)fsz - off < doc_bytes ? (size_t)fsz - off
                                                 : doc_bytes;
        size_t n_out = 0;
        double a = now();
        if (tsimd_tokenizer_encode_parallel_into(&t, buf + off, n, threads,
                                                 ebuf, ebound,
                                                 &n_out) != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        double b = now();
        enc_t += b - a;
        h = fold_u64(h, (uint64_t)n_out);
        size_t head = n_out < 4096 ? n_out : 4096;
        h = fold_ids(h, ebuf, head);
        if (n_out > head)
            h = fold_ids(h, ebuf + n_out - head, head);
        total += n_out;
        docs++;
        double c = now();
        size_t got = 0;
        int drc = tsimd_tokenizer_decode_parallel(&t, ebuf, n_out, 0, dbuf,
                                                  dbound, &got, threads);
        double d = now();
        dec_t += d - c;
        if (drc != 0 || got != n || memcmp(dbuf, buf + off, n) != 0)
            bad++;
        checked++;
    }
    double t1 = now();
    printf("input: %ld bytes in %zu docs of %zu bytes\n", fsz, docs, doc_bytes);
    printf("ids: %llu\n", total);
    printf("hash: 0x%016llx\n", (unsigned long long)h);
    printf("roundtrip: %s (%llu checked)\n", bad == 0 ? "ok" : "MISMATCH",
           checked);
    printf("encode: %.3f s = %.1f MB/s\n", enc_t,
           (double)fsz / 1e6 / enc_t);
    printf("decode: %.3f s = %.1f MB/s\n", dec_t,
           (double)fsz / 1e6 / dec_t);
    printf("docs (round trip): %.3f s = %.1f MB/s\n", t1 - t0,
           (double)fsz / 1e6 / (t1 - t0));

    static const char *prompts[8] = {
        "Summarize the following article in one sentence.",
        "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
        "The quick brown fox jumps over the lazy dog.",
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Explain the difference between a list and a tuple in Python.",
        "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY 2 DESC;",
        "Machine learning models are trained on large corpora of text.",
        "The recipe calls for 2 cups of flour, 1 tsp of salt, and 3 eggs.",
    };
    uint64_t ph = 0xcbf29ce484222325ULL;
    unsigned long long pids = 0;
    double p0 = now();
    long batch_n = 256;
    tsimd_input_span_t *spans = malloc((size_t)batch_n * sizeof *spans);
    if (!spans) return 1;
    for (long done = 0; done < prompt_calls; done += batch_n) {
        long n = prompt_calls - done < batch_n ? prompt_calls - done
                                               : batch_n;
        for (long i = 0; i < n; i++) {
            spans[i].data = (const uint8_t *)prompts[(done + i) % 8];
            spans[i].length = strlen(prompts[(done + i) % 8]);
        }
        tsimd_token_batch_t batch;
        memset(&batch, 0, sizeof batch);
        if (tsimd_tokenizer_encode_batch(&t, spans, (size_t)n, &batch,
                                         threads) != 0) {
            fprintf(stderr, "prompt batch failed\n");
            return 1;
        }
        for (size_t d = 0; d < n; d++) {
            size_t a = batch.offsets[d];
            size_t b = batch.offsets[d + 1];
            ph = fold_u64(ph, (uint64_t)(b - a));
            ph = fold_ids(ph, batch.ids + a, b - a);
            pids += b - a;
        }
        tsimd_token_batch_free(&batch);
    }
    free(spans);
    double p1 = now();
    printf("prompts: %ld calls in %.3f s = %.0f calls/s, %llu ids, "
           "hash 0x%016llx\n",
           prompt_calls, p1 - p0, prompt_calls / (p1 - p0), pids,
           (unsigned long long)ph);

    free(ebuf);
    free(dbuf);
    munmap(buf, (size_t)fsz);
    close(fd);
    tsimd_tokenizer_free(&t);
    return 0;
}
