#include <tokensim/tokenizer.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static long rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long size = 0, resident = 0;
    if (fscanf(f, "%ld %ld", &size, &resident) != 2) resident = -1;
    fclose(f);
    return resident < 0 ? -1 : resident * 4;
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
        argc > 5 ? (size_t)strtoull(argv[5], NULL, 10) : (size_t)(2u << 20);
    long prompt_calls = argc > 6 ? strtol(argv[6], NULL, 10) : 20000;
    int threads = argc > 7 ? (int)strtol(argv[7], NULL, 10) : 0;
    if (doc_bytes == 0) doc_bytes = (size_t)(2u << 20);
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

    FILE *f = fopen(argv[4], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[4]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    rewind(f);
    uint8_t *buf = malloc(doc_bytes);
    if (!buf) return 1;

    uint64_t h = 0xcbf29ce484222325ULL;
    unsigned long long total = 0, checked = 0, bad = 0;
    size_t docs = 0;
    long rss[5] = {rss_kb(), -1, -1, -1, -1};
    double t0 = now();
    for (;;) {
        size_t n = fread(buf, 1, doc_bytes, f);
        if (n == 0) break;
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode_parallel(&t, buf, n, threads, &seq) != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        h = fold_u64(h, (uint64_t)seq.count);
        size_t head = seq.count < 4096 ? seq.count : 4096;
        h = fold_ids(h, seq.ids, head);
        if (seq.count > head)
            h = fold_ids(h, seq.ids + seq.count - head, head);
        total += seq.count;
        docs++;
        if (docs % 64 == 0 || n < doc_bytes) {
            tsimd_byte_sequence_t back;
            memset(&back, 0, sizeof back);
            if (tsimd_tokenizer_decode(&t, seq.ids, seq.count, 0, &back) != 0 ||
                back.len != n || memcmp(back.bytes, buf, n) != 0)
                bad++;
            checked++;
            tsimd_byte_sequence_free(&back);
        }
        tsimd_token_sequence_free(&seq);
        long pos = ftell(f);
        if (fsz > 0 && pos > 0) {
            for (int q = 1; q < 5; q++)
                if (rss[q] < 0 && pos * 4 >= fsz * q) rss[q] = rss_kb();
        }
    }
    double t1 = now();
    for (int i = 0; i < 5; i++)
        if (rss[i] < 0) rss[i] = rss_kb();
    printf("input: %ld bytes in %zu docs of %zu bytes\n", fsz, docs, doc_bytes);
    printf("ids: %llu\n", total);
    printf("hash: 0x%016llx\n", (unsigned long long)h);
    printf("roundtrip: %s (%llu checked)\n", bad == 0 ? "ok" : "MISMATCH",
           checked);
    printf("docs: %.3f s = %.1f MB/s\n", t1 - t0,
           (double)fsz / 1e6 / (t1 - t0));
    printf("rss: %ld %ld %ld %ld %ld KB\n", rss[0], rss[1], rss[2], rss[3],
           rss[4]);

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
    for (long i = 0; i < prompt_calls; i++) {
        const char *p = prompts[i % 8];
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode_parallel(&t, (const uint8_t *)p, strlen(p),
                                            threads, &seq) != 0) {
            fprintf(stderr, "prompt encode failed\n");
            return 1;
        }
        ph = fold_u64(ph, (uint64_t)seq.count);
        ph = fold_ids(ph, seq.ids, seq.count);
        pids += seq.count;
        tsimd_token_sequence_free(&seq);
    }
    double p1 = now();
    printf("prompts: %ld calls in %.3f s = %.0f calls/s, %llu ids, "
           "hash 0x%016llx\n",
           prompt_calls, p1 - p0, prompt_calls / (p1 - p0), pids,
           (unsigned long long)ph);

    free(buf);
    fclose(f);
    tsimd_tokenizer_free(&t);
    return 0;
}
