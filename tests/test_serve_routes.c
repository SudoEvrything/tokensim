#include "tokenizer.h"
#include "pretokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define O200K TSIMD_SRC_DIR "/models/o200k_base.tiktoken"
#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"

typedef struct {
    uint32_t *ids;
    size_t count, cap;
} idbuf_t;

static int idbuf_push(idbuf_t *b, const uint32_t *ids, size_t n) {
    if (b->count + n > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->count + n) cap *= 2;
        uint32_t *p = realloc(b->ids, cap * sizeof(uint32_t));
        if (!p) return -1;
        b->ids = p;
        b->cap = cap;
    }
    memcpy(b->ids + b->count, ids, n * sizeof(uint32_t));
    b->count += n;
    return 0;
}

static int oracle_encode(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                         idbuf_t *out) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    if (tsimd_pretokenize_scalar_ref(t->scheme, data, len, &spans) != 0) {
        fprintf(stderr, "oracle: pretokenize failed\n");
        tsimd_span_array_free(&spans);
        return -1;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t->vocab.max_token_len) != 0) {
        tsimd_span_array_free(&spans);
        return -1;
    }
    for (size_t i = 0; i < spans.count; i++) {
        size_t s = spans.spans[i].start, e = spans.spans[i].end;
        if (e <= s || e > len) {
            fprintf(stderr, "oracle: bad span %zu..%zu\n", s, e);
            return -1;
        }
        ws.out_count = 0;
        if (tsimd_bpe_encode(&t->vocab, &ws, data + s, e - s) != 0) {
            fprintf(stderr, "oracle: bpe failed at span %zu\n", i);
            return -1;
        }
        if (idbuf_push(out, ws.out, ws.out_count) != 0) return -1;
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_span_array_free(&spans);
    return 0;
}

static int fails = 0;

static void check(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                  const char *name) {
    idbuf_t want = {0}, got = {0};
    if (oracle_encode(t, data, len, &want) != 0) {
        fprintf(stderr, "FAIL %s: oracle error\n", name);
        fails++;
        return;
    }

    for (int pass = 0; pass < 2; pass++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) {
            fprintf(stderr, "FAIL %s: encode error (pass %d)\n", name, pass);
            fails++;
            return;
        }
        if (seq.count != want.count ||
            (want.count != 0 &&
             memcmp(seq.ids, want.ids, want.count * sizeof(uint32_t)) != 0)) {
            size_t k = 0;
            while (k < seq.count && k < want.count && seq.ids[k] == want.ids[k])
                k++;
            fprintf(stderr,
                    "FAIL %s pass %d: %zu ids vs %zu, first divergence at "
                    "%zu (got %u, want %u)\n",
                    name, pass, seq.count, want.count, k,
                    k < seq.count ? seq.ids[k] : 0,
                    k < want.count ? want.ids[k] : 0);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    free(want.ids);
    free(got.ids);
}

int main(int argc, char **argv) {
    const char *scheme = "o200k";
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--scheme") == 0 && i + 1 < argc)
            scheme = argv[++i];
    const int o200k = strcmp(scheme, "o200k") == 0;

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof(t));
    if (tsimd_tokenizer_load(&t, o200k ? O200K : CL100K,
                             o200k ? TSIMD_PRETOK_O200K : TSIMD_PRETOK_CL100K,
                             NULL, 0) != 0) {
        fprintf(stderr, "cannot load %s\n", scheme);
        return 1;
    }

    static const char *cases[] = {
        "",
        "a",
        " ",
        "\n",
        "hello world, this is a plain sentence with punctuation.",
        "\0abc def\0",
        "aaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious",
        "internationalization internationalization",
        "1234567890123456789012345678901234567890",
        "a1b2c3d4e5f6g7h8i9j0",
        "don't I've we're they'll he'd",
        "path/to/file.txt // comment\r\n",
        "\r\n\r\n\r\n",
        "a//b///c\r\n/d",
        "caf\303\251 \303\251t\303\251 2024",
        "\346\227\245\346\234\254\350\252\236 \360\237\230\200 emoji",
        "\331\241" "23456789",
        "\303",
        "\377\376\375",
        "e\314\201cole\342\200\231s",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char name[64];
        snprintf(name, sizeof(name), "case%zu", i);
        check(&t, (const uint8_t *)cases[i], strlen(cases[i]), name);
    }

    {
        uint8_t buf[1024];
        for (int i = 0; i < 1024; i++) buf[i] = (uint8_t)(i * 7 + (i >> 3));
        check(&t, buf, sizeof(buf), "bytesweep");
    }

    for (size_t n = 60; n <= 72; n++) {
        uint8_t buf[80];
        for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)("abc def/.\r\n"[i % 10]);
        char name[32];
        snprintf(name, sizeof(name), "tail%zu", n);
        check(&t, buf, n, name);
    }

    {
        enum { N = 1u << 18 };
        uint8_t *buf = malloc(N);
        if (!buf) return 1;
        static const uint8_t alpha[] =
            " aabbccddeeffgghhiijjkk  0123456789 .,;:!?()-_/\\'\"\r\n\t"
            "\303\251\346\227\245\377\000";
        uint32_t s = 0x1234567u;
        for (size_t i = 0; i < N; i++) {
            s = s * 1103515245u + 12345u;
            buf[i] = alpha[(s >> 16) % (sizeof(alpha) - 1)];
        }
        check(&t, buf, N, "fuzz");
        free(buf);
    }

    tsimd_tokenizer_free(&t);
    if (fails) {
        printf("serve_routes %s: %d FAILURES\n", scheme, fails);
        return 1;
    }
    printf("serve_routes %s: all cases match the uncached oracle\n", scheme);
    return 0;
}
