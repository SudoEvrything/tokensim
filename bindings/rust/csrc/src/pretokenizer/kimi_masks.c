#include "internal.h"
#include "unicode.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t letter, upper, lower, digit, punct, ws, crlf, high, apos, con1,
        con2, con2b;
} kmask_t;

static inline uint64_t mm_pair(__m256i a, __m256i b) {
    return (uint64_t)(uint32_t)_mm256_movemask_epi8(a) |
           ((uint64_t)(uint32_t)_mm256_movemask_epi8(b) << 32);
}

static inline __m256i u8_lt(__m256i t, uint8_t w) {
    __m256i bias = _mm256_set1_epi8((char)0x80);
    return _mm256_cmpgt_epi8(_mm256_set1_epi8((char)(w ^ 0x80)),
                             _mm256_xor_si256(t, bias));
}

static void kimi_masks_load(const uint8_t *p, kmask_t *m) {
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)p);
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
    __m256i f0 = _mm256_or_si256(v0, _mm256_set1_epi8(0x20));
    __m256i f1 = _mm256_or_si256(v1, _mm256_set1_epi8(0x20));
    __m256i l0 = _mm256_sub_epi8(f0, _mm256_set1_epi8('a'));
    __m256i l1 = _mm256_sub_epi8(f1, _mm256_set1_epi8('a'));
    uint64_t letter = mm_pair(u8_lt(l0, 26), u8_lt(l1, 26));
    __m256i u0 = _mm256_sub_epi8(v0, _mm256_set1_epi8('A'));
    __m256i u1 = _mm256_sub_epi8(v1, _mm256_set1_epi8('A'));
    uint64_t upper = mm_pair(u8_lt(u0, 26), u8_lt(u1, 26));
    __m256i n0 = _mm256_sub_epi8(v0, _mm256_set1_epi8('0'));
    __m256i n1 = _mm256_sub_epi8(v1, _mm256_set1_epi8('0'));
    uint64_t digit = mm_pair(u8_lt(n0, 10), u8_lt(n1, 10));
    __m256i t0 = _mm256_sub_epi8(v0, _mm256_set1_epi8(0x09));
    __m256i t1 = _mm256_sub_epi8(v1, _mm256_set1_epi8(0x09));
    uint64_t tab = mm_pair(u8_lt(t0, 5), u8_lt(t1, 5));
    uint64_t space =
        mm_pair(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8(' ')),
                _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(' ')));
    uint64_t ws = tab | space;
    uint64_t crlf =
        mm_pair(_mm256_or_si256(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\r')),
                                _mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\n'))),
                _mm256_or_si256(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\r')),
                                _mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\n'))));
    uint64_t high = mm_pair(_mm256_cmpgt_epi8(_mm256_setzero_si256(), v0),
                            _mm256_cmpgt_epi8(_mm256_setzero_si256(), v1));
    m->apos = mm_pair(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\'')),
                      _mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\'')));
    {
        __m256i a0 = _mm256_or_si256(v0, _mm256_set1_epi8(0x20));
        __m256i a1 = _mm256_or_si256(v1, _mm256_set1_epi8(0x20));
        __m256i s10 = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(a0, _mm256_set1_epi8('s')),
                            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('t'))),
            _mm256_or_si256(_mm256_cmpeq_epi8(a0, _mm256_set1_epi8('m')),
                            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('d'))));
        __m256i s11 = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(a1, _mm256_set1_epi8('s')),
                            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('t'))),
            _mm256_or_si256(_mm256_cmpeq_epi8(a1, _mm256_set1_epi8('m')),
                            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('d'))));
        m->con1 = mm_pair(s10, s11);
        __m256i r10 = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(a0, _mm256_set1_epi8('r')),
                            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('v'))),
            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('l')));
        __m256i r11 = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(a1, _mm256_set1_epi8('r')),
                            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('v'))),
            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('l')));
        m->con2 = mm_pair(r10, r11);
        __m256i b10 = _mm256_or_si256(
            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('e')),
            _mm256_cmpeq_epi8(a0, _mm256_set1_epi8('l')));
        __m256i b11 = _mm256_or_si256(
            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('e')),
            _mm256_cmpeq_epi8(a1, _mm256_set1_epi8('l')));
        m->con2b = mm_pair(b10, b11);
    }
    m->letter = letter;
    m->upper = upper;
    m->lower = letter & ~upper;
    m->digit = digit;
    m->punct = ~(letter | digit | ws | high);
    m->ws = ws;
    m->crlf = crlf;
    m->high = high;
}

static size_t run_tail_end(const uint8_t *d, size_t n, size_t p, int which) {
    while (p < n) {
        uint8_t b = d[p];
        int in;
        if (b & 0x80) break;
        if (which == 0)
            in = ((b | 0x20) >= 'a' && (b | 0x20) <= 'z') ||
                 (b >= 'A' && b <= 'Z');
        else if (which == 1)
            in = b == ' ' || b == '\t' || b == '\n' || b == '\r' ||
                 b == 0x0b || b == 0x0c;
        else if (which == 2)
            in = b == '\r' || b == '\n';
        else if (which == 3)
            in = b >= '0' && b <= '9';
        else
            in = !(b == ' ' || b == '\t' || b == '\n' || b == '\r' ||
                   b == 0x0b || b == 0x0c) &&
                 !((b | 0x20) >= 'a' && (b | 0x20) <= 'z') &&
                 !(b >= 'A' && b <= 'Z') && !(b >= '0' && b <= '9');
        if (!in) break;
        p++;
    }
    return p;
}

static size_t run_end(const uint8_t *d, size_t n, size_t i, int which) {
    size_t p = i;
    while (p < n) {
        size_t avail = n - p;
        if (avail < 64) return run_tail_end(d, n, p, which);
        kmask_t m;
        kimi_masks_load(d + p, &m);
        uint64_t cmask = which == 0   ? m.letter
                         : which == 1 ? m.ws
                         : which == 2 ? m.crlf
                         : which == 3 ? m.digit
                                      : m.punct;
        if (!(cmask & 1)) return p;
        uint64_t inv = ~cmask;
        unsigned t = inv ? (unsigned)__builtin_ctzll(inv) : 64;
        if (t < 64) return p + t;
        p += 64;
    }
    return p;
}

static size_t word_end(const uint8_t *d, size_t n, size_t start) {
    size_t p = start;
    int prev_low = 0;
    while (p < n) {
        size_t avail = n - p;
        if (avail < 64) {
            while (p < n) {
                uint8_t b = d[p];
                if (b & 0x80) break;
                if (!((b | 0x20) >= 'a' && (b | 0x20) <= 'z') &&
                    !(b >= 'A' && b <= 'Z'))
                    break;
                if (b >= 'A' && b <= 'Z') {
                    if (prev_low) break;
                } else {
                    prev_low = 1;
                }
                p++;
            }
            return p;
        }
        kmask_t m;
        kimi_masks_load(d + p, &m);
        if (m.high) break;
        if (!(m.letter & 1)) break;
        uint64_t inv = ~m.letter;
        unsigned t = inv ? (unsigned)__builtin_ctzll(inv) : 64;
        uint64_t valid = t < 64 ? ((1ULL << t) - 1) : ~0ULL;
        uint64_t up_after_low = m.upper & (m.lower << 1) & valid;
        if (prev_low) up_after_low |= m.upper & valid & 1;
        if (up_after_low) return p + (size_t)__builtin_ctzll(up_after_low);
        if (t < 64) return p + t;
        prev_low = (int)((m.lower >> 63) & 1);
        p += 64;
    }
    return p;
}

static int contraction_len(const uint8_t *d, size_t n, size_t e) {
    if (e >= n || d[e] != '\'') return 0;
    if (e + 1 >= n) return 0;
    uint8_t c = d[e + 1];
    if (c == 's' || c == 'S' || c == 't' || c == 'T' || c == 'm' ||
        c == 'M' || c == 'd' || c == 'D')
        return 2;
    if (e + 2 >= n) return 0;
    uint8_t c2 = d[e + 2];
    if ((c == 'r' || c == 'R') && (c2 == 'e' || c2 == 'E')) return 3;
    if ((c == 'v' || c == 'V') && (c2 == 'e' || c2 == 'E')) return 3;
    if ((c == 'l' || c == 'L') && (c2 == 'l' || c2 == 'L')) return 3;
    return 0;
}

#define KCL_L 1
#define KCL_D 2
#define KCL_P 4
#define KCL_S 8
#define KCL_R 16

static size_t scalar_span(const uint8_t *d, size_t n, size_t i) {
    size_t e = tsimd_match_kimi(d, n, i);
    return e > i ? e : i + 1;
}

typedef struct {
    size_t wbase;
    kmask_t m;
} kwc_t;

static size_t kimi_next_span(const uint8_t *d, size_t n, size_t i,
                             kwc_t *wc) {
    if (d[i] & 0x80) return scalar_span(d, n, i);
    if (i + 1 < n && (d[i + 1] & 0x80)) return scalar_span(d, n, i);
    if (n - i < 72) return scalar_span(d, n, i);

    if (wc->wbase != i) {
        kimi_masks_load(d + i, &wc->m);
        wc->wbase = i;
    }
    kmask_t *m = &wc->m;

    if (m->letter & 1) {
        size_t e = word_end(d, n, i);
        if (e == i || (e < n && (d[e] & 0x80))) return scalar_span(d, n, i);
        return e + (size_t)contraction_len(d, n, e);
    }

    int crlf_here = (m->crlf & 1) != 0;
    if (!crlf_here && ((m->punct & 1) || (m->ws & 1))) {
        if (i + 1 < n && (m->letter >> 1) & 1) {
            size_t e = word_end(d, n, i + 1);
            if (e == i + 1 || (e < n && (d[e] & 0x80)))
                return scalar_span(d, n, i);
            return e + (size_t)contraction_len(d, n, e);
        }
        size_t r = run_end(d, n, i + 1, 4);
        if (r < n && (d[r] & 0x80)) return scalar_span(d, n, i);
        size_t ct = run_end(d, n, r, 2);
        if (ct > i + 1 || !(m->ws & 1)) return ct > i + 1 ? ct : i + 1;
    }

    if (m->digit & 1) {
        size_t r = run_end(d, n, i, 3);
        return i + 3 < r ? i + 3 : r;
    }

    size_t wsend = run_end(d, n, i, 1);
    size_t p = i;
    size_t last_crlf_end = 0;
    int found = 0;
    while (p < wsend) {
        size_t a2 = wsend - p;
        if (a2 < 64) {
            for (size_t k = p; k < wsend; k++)
                if (d[k] == '\r' || d[k] == '\n') {
                    last_crlf_end = k + 1;
                    found = 1;
                }
            break;
        }
        kmask_t m2;
        kimi_masks_load(d + p, &m2);
        uint64_t span_bits = a2 < 64 ? ((1ULL << a2) - 1) : ~0ULL;
        uint64_t cr = m2.crlf & span_bits;
        if (cr) {
            last_crlf_end = p + 64 - (size_t)__builtin_clzll(cr);
            found = 1;
        }
        p += a2 < 64 ? a2 : 64;
    }
    if (found) return last_crlf_end;
    if (wsend == n) return wsend;
    if (wsend > i + 1) return wsend - 1;
    return wsend;
}

static int kimi_masks_feed(const uint8_t *data, size_t len, tsimd_span_fn fn,
                           void *ud, tsimd_span_array_t *out) {
    size_t i = 0;
    kwc_t wc;
    wc.wbase = (size_t)-1;
    while (i < len) {
        size_t e = kimi_next_span(data, len, i, &wc);
        if (e <= i || e > len) return -1;
        if (fn) {
            if (fn(ud, i, e) != 0) return -1;
        } else if (out) {
            if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
                return -1;
        }
        i = e;
    }
    return 0;
}

int tsimd_pretokenize_kimi_masks(const uint8_t *data, size_t len,
                                 tsimd_span_array_t *out) {
    return kimi_masks_feed(data, len, NULL, NULL, out);
}

int tsimd_pretokenize_kimi_masks_ex(const uint8_t *data, size_t len,
                                    tsimd_span_fn fn, void *ud) {
    return kimi_masks_feed(data, len, fn, ud, NULL);
}
