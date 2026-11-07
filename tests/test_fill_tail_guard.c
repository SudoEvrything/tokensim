#include "pretokenizer.h"
#include "bpe.h"
#include "tokenizer.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*fill_fn_t)(const uint8_t *, size_t, void *,
                         tsimd_pcache_span_t *, size_t,
                         tsimd_pretok_cache_t *, int);

typedef struct {
    const char *name;
    fill_fn_t fill;
    tsimd_pretok_scheme_t scheme;
} scheme_t;

static const scheme_t SCHEMES[] = {
    { "o200k",    tsimd_o200k_mask_fill_keyed,    TSIMD_PRETOK_O200K },
    { "cl100k",   tsimd_cl100k_mask_fill_keyed,   TSIMD_PRETOK_CL100K },
    { "deepseek", tsimd_deepseek_mask_fill_keyed, TSIMD_PRETOK_DEEPSEEK },
    { "nemo",     tsimd_nemo_mask_fill_keyed,     TSIMD_PRETOK_NEMO },
    { "gpt2",     tsimd_gpt2_mask_fill_keyed,     TSIMD_PRETOK_GPT2 },
};
#define N_SCHEMES (sizeof SCHEMES / sizeof SCHEMES[0])

static int g_fail;

static void check_fill_bounds(const scheme_t *s, const char *label,
                              const uint8_t *in, size_t len, int twophase,
                              size_t cap) {
    tsimd_pcache_span_t *ents = malloc(cap * sizeof *ents);
    if (!ents) {
        fprintf(stderr, "FAIL: out of memory\n");
        exit(1);
    }
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    const char *path = twophase ? "twophase" : "fused";
    size_t calls = 0;
    for (;;) {
        int n = s->fill(in, len, &st, ents, cap, NULL, twophase);
        if (n < 0) {
            fprintf(stderr, "FAIL %s/%s %s: fill returned -1\n", s->name,
                    path, label);
            g_fail = 1;
            break;
        }
        for (int i = 0; i < n && !g_fail; i++) {
            const uint8_t *p = ents[i].ptr;
            size_t el = tsimd_pcache_span_len(&ents[i]);
            if (p < in || p > in + len || el == 0 || p + el > in + len) {
                fprintf(stderr,
                        "FAIL %s/%s cap %zu %s: entry %d at offset %zu "
                        "length %zu escapes input len %zu\n",
                        s->name, path, cap, label, i, (size_t)(p - in), el,
                        len);
                g_fail = 1;
            }
        }
        if (g_fail) break;
        if (st.pos > len) {
            fprintf(stderr, "FAIL %s/%s %s: state pos %zu past input len %zu\n",
                    s->name, path, label, st.pos, len);
            g_fail = 1;
            break;
        }
        if (st.pos >= len) break;
        if (n == 0) {
            fprintf(stderr, "FAIL %s/%s %s: no progress at pos %zu of %zu\n",
                    s->name, path, label, st.pos, len);
            g_fail = 1;
            break;
        }
        if (++calls > (1u << 20)) {
            fprintf(stderr, "FAIL %s/%s %s: did not terminate\n", s->name,
                    path, label);
            g_fail = 1;
            break;
        }
    }
    free(ents);
}

static void check_input(const char *label, const uint8_t *in, size_t len) {
    static const size_t caps[] = { 64, 128, 4096 };
    for (size_t si = 0; si < N_SCHEMES && !g_fail; si++)
        for (int tp = 0; tp <= 1 && !g_fail; tp++)
            for (size_t ci = 0;
                 ci < sizeof caps / sizeof caps[0] && !g_fail; ci++)
                check_fill_bounds(&SCHEMES[si], label, in, len, tp, caps[ci]);
}

#define TAILN 128
static void tail_fill(uint8_t *p) {
    const char *t = " abcdefghello";
    for (size_t i = 0; i < TAILN; i++) p[i] = (uint8_t)t[i % 14];
}

typedef void (*fill_bytes_fn)(uint8_t *p, size_t len, uint64_t seed);

static void f_d9(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    memset(p, 0xd9, len);
}
static void f_d9_nul(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    memset(p, 0xd9, len);
    if (len) p[len - 1] = 0x00;
}
static void f_letters(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    memset(p, 'a', len);
}
static void f_digits(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    memset(p, '1', len);
}
static void f_euro(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    for (size_t i = 0; i < len; i++) p[i] = "\xe2\x82\xac"[i % 3];
}
static void f_text(uint8_t *p, size_t len, uint64_t seed) {
    (void)seed;
    const char *t = "hello world\n";
    for (size_t i = 0; i < len; i++) p[i] = (uint8_t)t[i % 12];
}
static void f_rand(uint8_t *p, size_t len, uint64_t seed) {
    uint64_t r = seed;
    for (size_t i = 0; i < len; i++) {
        uint64_t v = tsimd_test_rand(&r);
        switch (v % 4) {
            case 0: p[i] = (uint8_t)" \t\n\r abcdefghello"[v % 17]; break;
            case 1: p[i] = (uint8_t)(v % 128); break;
            case 2: p[i] = (uint8_t)(0x80 + v % 128); break;
            default: p[i] = (uint8_t)"'\"\0/\r\n-"[v % 8]; break;
        }
    }
}

static void case_run(const char *label, fill_bytes_fn f, size_t len,
                     uint64_t seed) {
    uint8_t *in = malloc(len + TAILN);
    if (!in) {
        fprintf(stderr, "FAIL: out of memory\n");
        exit(1);
    }
    f(in, len, seed);
    tail_fill(in + len);
    check_input(label, in, len);
    free(in);
}

static const char *EXTRAS[] = {
    "ab", "abc", "abcd", " x", "th", "the", " t", " the",
    "er", "her", "in", "ing", "abab", "abababab",
};

static tsimd_tokenizer_t g_tok;
static int vocab_ready;

static int ensure_vocab(void) {
    if (vocab_ready) return 0;
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 14, &len);
    if (!buf) return -1;
    int rc = tsimd_vocab_load_buffer((const uint8_t *)buf, len, &g_tok.vocab);
    free(buf);
    if (rc != 0) return -1;
    if (tsimd_pretok_cache_init(&g_tok.cache, 4096) != 0) return -1;
    vocab_ready = 1;
    return 0;
}

static void check_roundtrip(const scheme_t *s, const char *label,
                            const uint8_t *in, size_t len) {
    if (ensure_vocab() != 0) {
        fprintf(stderr, "FAIL: vocab setup\n");
        exit(1);
    }
    g_tok.scheme = s->scheme;
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&g_tok, in, len, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode failed on %s\n", s->name, label);
        g_fail = 1;
        return;
    }
    tsimd_byte_sequence_t dec;
    if (tsimd_tokenizer_decode(&g_tok, seq.ids, seq.count, 0, &dec) != 0) {
        fprintf(stderr, "FAIL %s: decode failed on %s\n", s->name, label);
        g_fail = 1;
        tsimd_token_sequence_free(&seq);
        return;
    }
    if (dec.len != len || (len && memcmp(dec.bytes, in, len) != 0)) {
        fprintf(stderr, "FAIL %s: round-trip %s: len %zu -> %zu\n", s->name,
                label, len, dec.len);
        g_fail = 1;
    }
    tsimd_byte_sequence_free(&dec);
    tsimd_token_sequence_free(&seq);
}

int main(void) {

    {
        size_t len = 2973;
        uint8_t *in = malloc(len + TAILN);
        if (!in) return 1;
        memset(in, 0xd9, len);
        in[len - 1] = 0x00;
        tail_fill(in + len);
        check_input("d9x2972+NUL", in, len);
        for (size_t si = 0; si < N_SCHEMES && !g_fail; si++)
            check_roundtrip(&SCHEMES[si], "d9x2972+NUL", in, len);
        free(in);
    }

    static const size_t LENS[] = { 64, 65, 127, 128, 129, 2944, 2945,
                                   2972, 2973, 3007, 3008, 4096, 5000 };
    for (size_t li = 0; li < sizeof LENS / sizeof LENS[0] && !g_fail; li++) {
        size_t len = LENS[li];
        case_run("d9", f_d9, len, 0);
        case_run("d9+NUL", f_d9_nul, len, 0);
        case_run("letters", f_letters, len, 0);
        case_run("digits", f_digits, len, 0);
        case_run("euro", f_euro, len, 0);
        case_run("text", f_text, len, 0);
    }
    case_run("rand-8192", f_rand, 8192, 0x243F6A8885A308D3ull);
    case_run("rand-3000", f_rand, 3000, 0x9E3779B97F4A7C15ull);

    case_run("empty", f_d9, 0, 0);
    case_run("one", f_d9, 1, 0);

    if (g_fail) {
        fprintf(stderr, "test_fill_tail_guard: FAILED\n");
        return 1;
    }
    printf("test_fill_tail_guard: OK (all fills stayed inside the input)\n");
    return 0;
}
