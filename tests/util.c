#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void b64_encode(const uint8_t *in, size_t n, char *out) {
    size_t o = 0;
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16 | (uint32_t)in[i + 1] << 8 |
                     in[i + 2];
        out[o++] = B64[(v >> 18) & 63];
        out[o++] = B64[(v >> 12) & 63];
        out[o++] = B64[(v >> 6) & 63];
        out[o++] = B64[v & 63];
    }
    if (n - i == 1) {
        uint32_t v = (uint32_t)in[i] << 16;
        out[o++] = B64[(v >> 18) & 63];
        out[o++] = B64[(v >> 12) & 63];
        out[o++] = '=';
        out[o++] = '=';
    } else if (n - i == 2) {
        uint32_t v = (uint32_t)in[i] << 16 | (uint32_t)in[i + 1] << 8;
        out[o++] = B64[(v >> 18) & 63];
        out[o++] = B64[(v >> 12) & 63];
        out[o++] = B64[(v >> 6) & 63];
        out[o++] = '=';
    }
}

char *tsimd_test_make_vocab(const char **extras, size_t n_extras,
                           size_t *out_len) {

    size_t cap = 256 * 5 + 2;
    for (size_t i = 0; i < n_extras; i++) {
        size_t n = strlen(extras[i]);
        if (n > 180) return NULL;
        cap += (n + 2) / 3 * 4 + 1;
    }
    char *buf = malloc(cap);
    size_t o = 0;
    char line[256];

    for (int b = 0; b < 256; b++) {
        uint8_t byte = (uint8_t)b;
        b64_encode(&byte, 1, line);
        line[4] = '\n';
        memcpy(buf + o, line, 5);
        o += 5;
    }
    for (size_t i = 0; i < n_extras; i++) {
        size_t n = strlen(extras[i]);
        b64_encode((const uint8_t *)extras[i], n, line);
        size_t enc = (n + 2) / 3 * 4;
        line[enc] = '\n';
        memcpy(buf + o, line, enc + 1);
        o += enc + 1;
    }
    buf[o] = 0;
    *out_len = o;
    return buf;
}

long tsimd_test_unescape(const char *s, size_t n, uint8_t *out, size_t cap,
                        size_t *out_len) {
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t c = (uint8_t)s[i];
        if (c == '\\') {
            if (i + 1 >= n) return -1;
            char e = s[++i];
            switch (e) {
                case '\\': c = '\\'; break;
                case '0':  c = '\0'; break;
                case 't':  c = '\t'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                default: return -1;
            }
        }
        if (o >= cap) return -1;
        out[o++] = c;
    }
    *out_len = o;
    return 0;
}

char *tsimd_test_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = 0;
    *out_len = (size_t)sz;
    return buf;
}

uint64_t tsimd_test_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
