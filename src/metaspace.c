#include "unicode.h"
#include "spm_charmap.h"
#include "spm_seqmap.h"

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif

#define SPM_FLUSH()                                                            \
    do {                                                                       \
        if (pending && have_content) {                                         \
            if (!emitted) led_space = 1;                                       \
            out[n] = 0xE2; out[n + 1] = 0x96; out[n + 2] = 0x81;               \
            n += 3;                                                            \
            emitted = 1;                                                       \
        }                                                                      \
        pending = 0;                                                           \
    } while (0)

static inline int ms_step(const uint8_t *in, size_t len, size_t *ip,
                          uint8_t *out, size_t *np, size_t cap) {
    size_t i = *ip;
    size_t n = *np;
    if (in[i] == ' ') {
        if (n + 3 > cap) return -1;
        out[n++] = 0xE2;
        out[n++] = 0x96;
        out[n++] = 0x81;
        i++;
        *ip = i;
        *np = n;
        return 0;
    }
    if (in[i] < 0x80) {
        if (n + 1 > cap) return -1;
        out[n++] = in[i++];
        *ip = i;
        *np = n;
        return 0;
    }
    uint8_t b = in[i];
    size_t need = 0;
    if (b >= 0xC2 && b <= 0xDF) need = 1;
    else if (b >= 0xE0 && b <= 0xEF) need = 2;
    else if (b >= 0xF0 && b <= 0xF4) need = 3;
    size_t adv = 1;
    int valid = 0;
    if (need > 0) {
        size_t k = 0;
        while (k < need && i + 1 + k < len && in[i + 1 + k] >= 0x80 &&
               in[i + 1 + k] <= 0xBF)
            k++;
        adv = 1 + k;
        if (k == need) {
            valid = 1;
            if (b == 0xE0 && in[i + 1] < 0xA0) valid = 0;
            if (b == 0xED && in[i + 1] > 0x9F) valid = 0;
            if (b == 0xF0 && in[i + 1] < 0x90) valid = 0;
            if (b == 0xF4 && in[i + 1] > 0x8F) valid = 0;
        }
    }
    if (valid) {
        if (n + adv > cap) return -1;
        memcpy(out + n, in + i, adv);
        n += adv;
        i += adv;
    } else {
        if (n + 3 > cap) return -1;
        out[n++] = 0xEF;
        out[n++] = 0xBF;
        out[n++] = 0xBD;
        i += adv;
    }
    *ip = i;
    *np = n;
    return 0;
}

size_t tsimd_metaspace_ex(const uint8_t *in, size_t len, uint8_t *out,
                          size_t cap, int dummy_prefix) {
    size_t n = 0;
    size_t i = 0;
    if (dummy_prefix && len > 0) {
        if (n + 3 > cap) return (size_t)-1;
        out[n++] = 0xE2;
        out[n++] = 0x96;
        out[n++] = 0x81;
    }
#if defined(__x86_64__) || defined(__i386__)
    const __m128i sp = _mm_set1_epi8(' ');
    const __m128i him = _mm_set1_epi8((char)0x80);
    while (i + 16 <= len) {
        __m128i v =
            _mm_loadu_si128((const __m128i *)(const void *)(in + i));
        unsigned msp = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(v, sp));
        unsigned mhi = (unsigned)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_and_si128(v, him), him));
        if ((msp | mhi) == 0) {
            if (n + 16 > cap) return (size_t)-1;
            _mm_storeu_si128((__m128i *)(void *)(out + n), v);
            n += 16;
            i += 16;
            continue;
        }
        size_t blk_end = i + 16;
        if (mhi == 0) {
            unsigned pc = (unsigned)__builtin_popcount(msp);
            if (i + 32 <= len && n + 16 + 2u * pc + 15 <= cap) {
                unsigned rest = msp;
                unsigned prev = 0;
                while (rest) {
                    unsigned k = (unsigned)__builtin_ctz(rest);
                    if (k > prev) {
                        _mm_storeu_si128(
                            (__m128i *)(void *)(out + n),
                            _mm_loadu_si128((const __m128i *)(const void *)(
                                in + i + prev)));
                        n += k - prev;
                    }
                    out[n] = 0xE2;
                    out[n + 1] = 0x96;
                    out[n + 2] = 0x81;
                    n += 3;
                    prev = k + 1;
                    rest &= rest - 1;
                }
                if (prev < 16) {
                    _mm_storeu_si128(
                        (__m128i *)(void *)(out + n),
                        _mm_loadu_si128(
                            (const __m128i *)(const void *)(in + i + prev)));
                    n += 16 - prev;
                }
                i = blk_end;
                continue;
            }
        }
        while (i < blk_end)
            if (ms_step(in, len, &i, out, &n, cap) != 0)
                return (size_t)-1;
    }
#endif
    while (i < len)
        if (ms_step(in, len, &i, out, &n, cap) != 0)
            return (size_t)-1;
    return n;
}

static const tsimd_spm_map_t *spm_cp_rule(uint32_t cp) {
    int lo, hi;
    if (cp < 0x10000) {
        if (!(tsimd_spm_charmap_bits[cp >> 3] & (1u << (cp & 7))))
            return NULL;
    } else {
        lo = 0;
        hi = tsimd_spm_charmap_high_n - 1;
        int found = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint32_t v = tsimd_spm_charmap_high[mid];
            if (v < cp) lo = mid + 1;
            else if (v > cp) hi = mid - 1;
            else { found = 1; break; }
        }
        if (!found) return NULL;
    }
    lo = 0;
    hi = tsimd_spm_charmap_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const tsimd_spm_map_t *e = &tsimd_spm_charmap[mid];
        if (e->cp < cp) lo = mid + 1;
        else if (e->cp > cp) hi = mid - 1;
        else return e;
    }
    return NULL;
}

static uint32_t spm_canon(uint32_t cp) {
    const tsimd_spm_map_t *e = spm_cp_rule(cp);
    if (e && e->len == 1) return tsimd_spm_charmap_str[e->off];
    return cp;
}

static const tsimd_spm_seq_t *spm_seq_rule(const uint32_t *key, size_t n) {
    int lo = 0, hi = tsimd_spm_seq_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const tsimd_spm_seq_t *e = &tsimd_spm_seq[mid];
        size_t m = e->klen < n ? e->klen : n;
        int c = 0;
        for (size_t q = 0; q < m; q++) {
            uint32_t a = tsimd_spm_seq_str[e->off + q];
            uint32_t b = key[q];
            if (a != b) { c = a < b ? -1 : 1; break; }
        }
        if (c == 0 && e->klen != n) c = e->klen < n ? -1 : 1;
        if (c == 0) return e;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

static int spm_is_space(uint32_t cp, tsimd_char_class_t cls) {
    if (cp == 0x85) return 0;
    if (cls == TSIMD_CLASS_SPACE) return 1;
    return (cp >= 0x200B && cp <= 0x200F) || cp == 0xFEFF ||
           cp == 0xFFFD || cp == 0x2581;
}

static uint64_t g_ascii_ruled[2];
static _Atomic int g_ascii_ruled_ready = 0;

static inline __attribute__((always_inline)) int spm_ascii_ruled(uint32_t cp) {
    if (atomic_load_explicit(&g_ascii_ruled_ready, memory_order_acquire) ==
        0) {
        uint64_t tmp[2] = {0, 0};
        for (uint32_t q = 0; q < 0x80; q++)
            if (tsimd_spm_charmap_bits[q >> 3] & (1u << (q & 7)))
                tmp[q >> 6] |= 1ull << (q & 63);
        g_ascii_ruled[0] = tmp[0];
        g_ascii_ruled[1] = tmp[1];
        atomic_store_explicit(&g_ascii_ruled_ready, 1, memory_order_release);
    }
    return (int)((g_ascii_ruled[cp >> 6] >> (cp & 63)) & 1u);
}

static inline __attribute__((always_inline)) int spm_plain_ascii(
    const uint8_t *in, size_t len, size_t i) {
    uint8_t b = in[i];
    if (b < 0x21 || b > 0x7E) return 0;
    if (i + 1 < len && in[i + 1] >= 0xC0) return 0;
    return !spm_ascii_ruled(b);
}

size_t tsimd_spm_normalize(const uint8_t *in, size_t len, uint8_t *out,
                           size_t cap, int dummy_prefix, int remove_extra) {
    size_t n = 0;
    size_t poff = 0;
    int emitted = 0;
    int led_space = 0;

    {
        int pending = 0;
        int have_content = 0;
        size_t i = 0;
        if (remove_extra) {
            poff = dummy_prefix ? 3 : 0;
            n = poff;
            if (n > cap) return (size_t)-1;
        }
        if (!remove_extra) {

            return tsimd_metaspace_ex(in, len, out, cap, dummy_prefix);
        }
        while (i < len) {
            if (spm_plain_ascii(in, len, i)) {
                if (n + 4 > cap) return (size_t)-1;
                SPM_FLUSH();
                have_content = 1;
                emitted = 1;
                out[n++] = in[i];
                i++;
                continue;
            }
            if (in[i] == ' ' && !spm_ascii_ruled(' ') &&
                (i + 1 >= len || in[i + 1] < 0xC0)) {
                pending = 1;
                i++;
                continue;
            }
            uint32_t cps[TSIMD_SPM_SEQ_MAXKEY];
            uint8_t blen[TSIMD_SPM_SEQ_MAXKEY];
            size_t cpn = 0, win = 0;
            tsimd_char_class_t cls0 = TSIMD_CLASS_OTHER;
            int bad = 0;
            while (cpn < TSIMD_SPM_SEQ_MAXKEY && i + win < len) {
                uint32_t cp = 0;
                tsimd_char_class_t cls = TSIMD_CLASS_OTHER;
                size_t a;
                uint8_t b0 = in[i + win];
                if (b0 < 0x80) {
                    cp = b0;
                    a = 1;
                    if (b0 == ' ' || (b0 >= 0x09 && b0 <= 0x0D))
                        cls = TSIMD_CLASS_SPACE;
                } else {
                    a = tsimd_utf8_next(in + i + win, len - i - win, &cp,
                                        &cls);
                }
                if (a == 0) break;
                if (cpn == 0) {
                    cls0 = cls;

                    if (a == 1 && cp >= 0x80) { bad = 1; break; }
                    cps[0] = cp;
                    blen[0] = (uint8_t)a;
                    cpn = 1;
                    win = a;

                    if (i + a >= len || in[i + a] < 0xC0) break;
                    continue;
                }
                cps[cpn] = cp;
                blen[cpn] = (uint8_t)a;
                cpn++;
                win += a;
            }
            if (bad || cpn == 0) {
                if (n + 8 > cap) return (size_t)-1;
                SPM_FLUSH();
                have_content = 1;
                emitted = 1;
                out[n++] = 0xEF;
                out[n++] = 0xBF;
                out[n++] = 0xBD;
                i += 1;
                continue;
            }

            const uint32_t *tgt = NULL;
            size_t tlen = 0, adv = blen[0];
            int ruled = 0;

            if (cpn >= 2 && cps[1] >= 0x300) {
                for (size_t k = cpn; k >= 2; k--) {
                    uint32_t key[TSIMD_SPM_SEQ_MAXKEY];
                    for (size_t q = 0; q < k; q++) key[q] = spm_canon(cps[q]);
                    const tsimd_spm_seq_t *e = spm_seq_rule(key, k);
                    if (e) {
                        tgt = &tsimd_spm_seq_str[e->off + e->klen];
                        tlen = e->tlen;
                        adv = 0;
                        for (size_t q = 0; q < k; q++) adv += blen[q];
                        ruled = 1;
                        break;
                    }
                }
            }
            if (!ruled) {
                const tsimd_spm_map_t *e = spm_cp_rule(cps[0]);
                if (e) {
                    tgt = &tsimd_spm_charmap_str[e->off];
                    tlen = e->len;
                    ruled = 1;
                }
            }
            if (ruled && tlen == 0) {

                i += adv;
                continue;
            }
            if (ruled) {
                if (n + tlen * 4 + 7 > cap) return (size_t)-1;

                const int maps_to_space = tgt[0] == ' ';
                if (!maps_to_space) SPM_FLUSH();
                have_content = 1;
                size_t q0 = 0;
                if (maps_to_space) {

                    pending = 1;
                    q0 = 1;
                }
                for (size_t q = q0; q < tlen; q++) {
                    if (q == 1) SPM_FLUSH();
                    emitted = 1;
                    uint32_t c = tgt[q];
                    if (c == ' ') {
                        out[n++] = 0xE2;
                        out[n++] = 0x96;
                        out[n++] = 0x81;
                    } else if (c < 0x80) {
                        out[n++] = (uint8_t)c;
                    } else if (c < 0x800) {
                        out[n++] = (uint8_t)(0xC0 | (c >> 6));
                        out[n++] = (uint8_t)(0x80 | (c & 0x3F));
                    } else if (c < 0x10000) {
                        out[n++] = (uint8_t)(0xE0 | (c >> 12));
                        out[n++] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
                        out[n++] = (uint8_t)(0x80 | (c & 0x3F));
                    } else {
                        out[n++] = (uint8_t)(0xF0 | (c >> 18));
                        out[n++] = (uint8_t)(0x80 | ((c >> 12) & 0x3F));
                        out[n++] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
                        out[n++] = (uint8_t)(0x80 | (c & 0x3F));
                    }
                }
                i += adv;
                continue;
            }

            if (spm_is_space(cps[0], cls0)) {
                pending = 1;
            } else {
                if (n + adv + 3 > cap) return (size_t)-1;
                SPM_FLUSH();
                have_content = 1;
                emitted = 1;
                if (adv == 1) out[n] = in[i];
                else memcpy(out + n, in + i, adv);
                n += adv;
            }
            i += adv;
        }
    }
    if (n == 0) return 0;

    (void)dummy_prefix;
    if (n == poff) return 0;
    if (poff == 0) return n;
    if (!led_space) {
        out[0] = 0xE2; out[1] = 0x96; out[2] = 0x81;
        return n;
    }
    memmove(out, out + poff, n - poff);
    return n - poff;
}

size_t tsimd_metaspace(const uint8_t *in, size_t len, uint8_t *out,
                       size_t cap) {
    return tsimd_metaspace_ex(in, len, out, cap, 1);
}
