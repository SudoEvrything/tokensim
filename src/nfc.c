#include "nfc_internal.h"
#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif
#include "unicode.h"
#include "allocator.h"

#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define NFC_SBASE  0xAC00u
#define NFC_SCOUNT 11172u
#define NFC_LBASE  0x1100u
#define NFC_VBASE  0x1161u
#define NFC_TBASE  0x11A7u
#define NFC_TCOUNT 28u
#define NFC_NCOUNT 588u
#define NFC_LCOUNT 19u
#define NFC_VCOUNT 21u
#define NFC_INVALID 0x110000u

static const tsimd_nfc_decomp_t *decomp_find(uint32_t cp,
                                              const tsimd_nfc_decomp_t *tab,
                                              size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tab[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n && tab[lo].cp == cp)
        return &tab[lo];
    return NULL;
}

static inline uint8_t ccc_direct(uint32_t cp) {
    if (cp >= 0x110000) return 0;
    uint16_t off = tsimd_nfc_ccc_idx[cp >> 8];
    if (!off) return 0;
    return tsimd_nfc_ccc_blk[off + (cp & 0xFF)];
}

uint8_t tsimd_nfc_ccc_lookup(uint32_t cp) {
    return ccc_direct(cp);
}

static uint32_t comp_find(uint32_t a, uint32_t b) {
    size_t lo = 0, hi = tsimd_nfc_comp_n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const tsimd_nfc_pair_t *p = &tsimd_nfc_comp[mid];
        if (p->a < a || (p->a == a && p->b < b)) lo = mid + 1;
        else hi = mid;
    }
    if (lo < tsimd_nfc_comp_n && tsimd_nfc_comp[lo].a == a &&
        tsimd_nfc_comp[lo].b == b)
        return tsimd_nfc_comp[lo].c;
    return 0;
}

static uint32_t hangul_compose(uint32_t a, uint32_t b) {
    if (a >= NFC_SBASE && a < NFC_SBASE + NFC_SCOUNT) {
        uint32_t s = a - NFC_SBASE;
        if (s % NFC_TCOUNT == 0 && b >= NFC_VBASE && b < NFC_VBASE + NFC_VCOUNT)
            return a + (b - NFC_VBASE) * NFC_TCOUNT;
        if (s % NFC_TCOUNT == 0 && b > NFC_TBASE && b < NFC_TBASE + NFC_TCOUNT)
            return a + (b - NFC_TBASE);
        return 0;
    }
    if (b >= NFC_VBASE && b < NFC_VBASE + NFC_VCOUNT &&
        a >= NFC_LBASE && a < NFC_LBASE + NFC_LCOUNT)
        return NFC_SBASE + (a - NFC_LBASE) * NFC_NCOUNT +
               (b - NFC_VBASE) * NFC_TCOUNT;
    return 0;
}

static size_t utf8_put(uint32_t cp, uint8_t *out, size_t cap) {
    if (cp < 0x80u) {
        if (cap < 1) return 0;
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800u) {
        if (cap < 2) return 0;
        out[0] = (uint8_t)(0xC0u | (cp >> 6));
        out[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        if (cap < 3) return 0;
        out[0] = (uint8_t)(0xE0u | (cp >> 12));
        out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    if (cap < 4) return 0;
    out[0] = (uint8_t)(0xF0u | (cp >> 18));
    out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4;
}

static inline uint32_t nfc_mm_hi(const uint8_t *p);

int tsimd_nfc_may_change(const uint8_t *in, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        if (nfc_mm_hi(in + i)) return 1;
    for (; i < len; i++)
        if (in[i] >= 0x80) return 1;
    return 0;
}

int tsimd_nfkc_may_change(const uint8_t *in, size_t len) {

    return tsimd_nfc_may_change(in, len);
}

static inline uint32_t nfc_mm_hi(const uint8_t *p) {
#if defined(__x86_64__) || defined(__i386__)
    __m128i v = _mm_loadu_si128((const __m128i *)(const void *)p);
    return (uint32_t)_mm_movemask_epi8(v);
#else
    uint32_t mk = 0;
    for (int k = 0; k < 16; k++) mk |= (uint32_t)(p[k] >= 0x80) << k;
    return mk;
#endif
}

#define TSIMD_NFC_TL_MAX (16u << 20)

static _Thread_local uint32_t *tl_nfc_buf;
static _Thread_local size_t tl_nfc_cap;

void tsimd_nfc_tls_release(void) {
    free(tl_nfc_buf);
    tl_nfc_buf = NULL;
    tl_nfc_cap = 0;
}

static uint32_t *nfc_scratch_acquire(size_t need_cps, int *free_on_exit) {
    *free_on_exit = 0;
    if (need_cps <= tl_nfc_cap) return tl_nfc_buf;
    uint32_t *nb = (uint32_t *)malloc(need_cps * sizeof(uint32_t));
    if (!nb) return NULL;
    if (need_cps > TSIMD_NFC_TL_MAX) {
        *free_on_exit = 1;
        return nb;
    }
    free(tl_nfc_buf);
    tl_nfc_buf = nb;
    tl_nfc_cap = need_cps;
    return nb;
}

static size_t nfc_count_hi(const uint8_t *in, size_t len) {
    size_t hi = 0, i = 0;
    for (; i + 16 <= len; i += 16) {
#if defined(__x86_64__) || defined(__i386__)
        hi += (size_t)__builtin_popcount(nfc_mm_hi(in + i));
#else
        uint32_t mk = nfc_mm_hi(in + i);
        while (mk) {
            hi++;
            mk &= mk - 1;
        }
#endif
    }
    for (; i < len; i++)
        if (in[i] >= 0x80) hi++;
    return hi;
}

static size_t nfc_scan_ascii(const uint8_t *in, size_t len, size_t i) {
    for (; i + 16 <= len; i += 16) {
        uint32_t mk = nfc_mm_hi(in + i);
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
    while (i < len && in[i] < 0x80) i++;
    return i;
}

static size_t nfc_scan_hi(const uint8_t *in, size_t len, size_t i) {
    for (; i + 16 <= len; i += 16) {
        uint32_t mk = nfc_mm_hi(in + i);
        if (mk != 0xFFFFu) return i + (size_t)__builtin_ctz((~mk) & 0xFFFFu);
    }
    while (i < len && in[i] >= 0x80) i++;
    return i;
}

static size_t norm_impl(const uint8_t *in, size_t len, uint8_t *out,
                        size_t cap, int compat, tsimd_nfc_run_t *runs,
                        size_t max_runs) {
    size_t nr = 0;

    if (!tsimd_nfc_may_change(in, len)) {
        if (len > cap) return (size_t)-1;
        if (len > 0) memcpy(out, in, len);
        return len;
    }

    size_t hi = nfc_count_hi(in, len);
    size_t cap_cps = len + (compat ? tsimd_nfkc_worst : 3) * hi + 4;
    int buf_free_on_exit = 0;
    uint32_t *buf = nfc_scratch_acquire(cap_cps, &buf_free_on_exit);
    if (!buf) return (size_t)-1;

    const tsimd_nfc_decomp_t *tab = compat ? tsimd_nfkc_decomp
                                           : tsimd_nfc_decomp;
    size_t tab_n = compat ? tsimd_nfkc_decomp_n : tsimd_nfc_decomp_n;

    const uint32_t *cps_blob = compat ? tsimd_nfkc_decomp_cps
                                      : tsimd_nfc_decomp_cps;

    size_t o = 0;
    size_t i = 0;
    while (i < len) {
        size_t lim = nfc_scan_ascii(in, len, i);
        size_t aend = (lim == len) ? lim : (lim > i ? lim - 1 : i);
        if (runs == NULL) {
            if (o + (aend - i) > cap) {
                if (buf_free_on_exit) free(buf);
                return (size_t)-1;
            }
            if (out && aend > i) memcpy(out + o, in + i, aend - i);
            o += aend - i;
        }
        if (lim == len) break;

        size_t nend = nfc_scan_hi(in, len, lim);

    size_t m = 0;
    i = aend;
    while (i < nend) {
        uint32_t cp;
        size_t adv = tsimd_utf8_decode(in + i, len - i, &cp);
        i += adv;
        if (cp < 0x80u) {
            buf[m++] = cp;
            continue;
        }
        if (adv == 1) {
            buf[m++] = NFC_INVALID + cp;
            continue;
        }
        if (cp >= NFC_SBASE && cp < NFC_SBASE + NFC_SCOUNT) {
            uint32_t s = cp - NFC_SBASE;
            buf[m++] = NFC_LBASE + s / NFC_NCOUNT;
            buf[m++] = NFC_VBASE + (s % NFC_NCOUNT) / NFC_TCOUNT;
            uint32_t t = s % NFC_TCOUNT;
            if (t != 0) buf[m++] = NFC_TBASE + t;
            continue;
        }
        const tsimd_nfc_decomp_t *d = decomp_find(cp, tab, tab_n);
        if (!d) {
            buf[m++] = cp;
            continue;
        }
        for (uint32_t k = 0; k < d->len; k++)
            buf[m++] = cps_blob[d->off + k];
    }

    for (size_t a = 1; a < m; a++) {
        uint8_t ck = ccc_direct(buf[a]);
        if (ck == 0) continue;
        size_t j = a;
        while (j > 0) {
            uint8_t pv = ccc_direct(buf[j - 1]);
            if (pv == 0 || pv <= ck) break;
            uint32_t t = buf[j - 1];
            buf[j - 1] = buf[j];
            buf[j] = t;
            j--;
        }
    }

    size_t w = 0;
    size_t starter = SIZE_MAX;
    uint8_t last_ccc = 0;
    for (size_t r = 0; r < m; r++) {
        uint32_t cp = buf[r];
        if (starter != SIZE_MAX) {
            uint8_t cc = ccc_direct(cp);
            if (last_ccc == 0 || last_ccc < cc) {
                uint32_t c = hangul_compose(buf[starter], cp);
                if (c == 0) c = comp_find(buf[starter], cp);
                if (c != 0) {
                    buf[starter] = c;

                    continue;
                }
            }
        }
        buf[w] = cp;
        w++;
        uint8_t cc = ccc_direct(cp);
        if (cc == 0) {
            starter = w - 1;
            last_ccc = 0;
        } else {
            last_ccc = cc;
        }
    }

    size_t o_unit = o;
    for (size_t q = 0; q < w; q++) {
        uint32_t cp = buf[q];
        if (cp >= NFC_INVALID) {
            if (o + 1 > cap) {
                if (buf_free_on_exit) free(buf);
                return (size_t)-1;
            }
            if (out) out[o] = (uint8_t)(cp - NFC_INVALID);
            o++;
            continue;
        }
        size_t adv = utf8_put(cp, out ? out + o : NULL, cap - o);
        if (adv == 0) {
            if (buf_free_on_exit) free(buf);
            return (size_t)-1;
        }
        o += adv;
    }
        if (runs) {
            if (nr >= max_runs) {
                if (buf_free_on_exit) free(buf);
                return (size_t)-1;
            }
            runs[nr].in_off = (uint32_t)aend;
            runs[nr].in_len = (uint32_t)(nend - aend);
            runs[nr].out_off = (uint32_t)o_unit;
            runs[nr].out_len = (uint32_t)(o - o_unit);
            nr++;
        }
        i = nend;
    }
    if (buf_free_on_exit) free(buf);
    return runs ? nr : o;
}

size_t tsimd_nfc_normalize(const uint8_t *in, size_t len, uint8_t *out,
                           size_t cap) {
    return norm_impl(in, len, out, cap, 0, NULL, 0);
}

size_t tsimd_nfkc_normalize(const uint8_t *in, size_t len, uint8_t *out,
                            size_t cap) {
    return norm_impl(in, len, out, cap, 1, NULL, 0);
}

size_t tsimd_nfc_normalize_runs(const uint8_t *in, size_t len, uint8_t *out,
                                size_t cap, tsimd_nfc_run_t *runs,
                                size_t max_runs) {
    if (len == 0) return 0;
    if (!tsimd_nfc_may_change(in, len)) return 0;
    return norm_impl(in, len, out, cap, 0, runs, max_runs);
}

size_t tsimd_nfkc_normalize_runs(const uint8_t *in, size_t len, uint8_t *out,
                                 size_t cap, tsimd_nfc_run_t *runs,
                                 size_t max_runs) {
    if (len == 0) return 0;
    if (!tsimd_nfc_may_change(in, len)) return 0;
    return norm_impl(in, len, out, cap, 1, runs, max_runs);
}
