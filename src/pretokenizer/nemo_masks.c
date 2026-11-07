#include "pretokenizer.h"
#include "internal.h"
#include "unicode.h"
#include "bpe.h"

#include <immintrin.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t smear_up(uint64_t seed, uint64_t within) {
    uint64_t a = seed, m = within;
    for (unsigned sh = 1; sh < 64; sh <<= 1) {
        a |= (a << sh) & m;
        m &= m << sh;
    }
    return a;
}

static inline uint64_t digit_run_splits3(uint64_t d) {
    uint64_t b = d & ~(d << 1);
    uint64_t c = d & (d >> 1) & (d >> 2) & (d >> 3);
    for (unsigned sh = 3; sh < 64; sh <<= 1) {
        b |= (b & c) << sh;
        c &= c >> sh;
    }
    return b;
}

static inline uint64_t mm_pair(__m256i a, __m256i b) {
    return (uint32_t)_mm256_movemask_epi8(a) |
           ((uint64_t)(uint32_t)_mm256_movemask_epi8(b) << 32);
}

static inline __m256i u8lt(__m256i a, __m256i b) {
    __m256i bias = _mm256_set1_epi8((char)0x80);
    return _mm256_cmpgt_epi8(_mm256_xor_si256(b, bias),
                             _mm256_xor_si256(a, bias));
}

typedef struct {
    uint64_t l;
    uint64_t up;
    uint64_t d;
    uint64_t s;
    uint64_t wt;
    uint64_t n;
    uint64_t hi;
    uint64_t ap;
    uint64_t sl;
} amask_t;

static void ascii_masks_64(const uint8_t *p, amask_t *m) {
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)p);
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
    __m256i lc = _mm256_set1_epi8('a');
    __m256i l26 = _mm256_set1_epi8(26);
    __m256i lo0 = _mm256_or_si256(v0, _mm256_set1_epi8(0x20));
    __m256i lo1 = _mm256_or_si256(v1, _mm256_set1_epi8(0x20));
    m->l = mm_pair(u8lt(_mm256_sub_epi8(lo0, lc), l26),
                   u8lt(_mm256_sub_epi8(lo1, lc), l26));
    m->up = mm_pair(
        _mm256_and_si256(_mm256_cmpgt_epi8(v0, _mm256_set1_epi8(0x40)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8(0x5B), v0)),
        _mm256_and_si256(_mm256_cmpgt_epi8(v1, _mm256_set1_epi8(0x40)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8(0x5B), v1)));
    m->d = mm_pair(
        _mm256_and_si256(_mm256_cmpgt_epi8(v0, _mm256_set1_epi8(0x2F)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8(0x3A), v0)),
        _mm256_and_si256(_mm256_cmpgt_epi8(v1, _mm256_set1_epi8(0x2F)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8(0x3A), v1)));
    m->s = mm_pair(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8(' ')),
                   _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(' ')));
    __m256i n0 = _mm256_or_si256(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\r')),
                                 _mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\n')));
    __m256i n1 = _mm256_or_si256(_mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\r')),
                                 _mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\n')));
    m->n = mm_pair(n0, n1);

    m->wt = mm_pair(
        _mm256_andnot_si256(n0,
            _mm256_and_si256(_mm256_cmpgt_epi8(v0, _mm256_set1_epi8(8)),
                             _mm256_cmpgt_epi8(_mm256_set1_epi8(14), v0))),
        _mm256_andnot_si256(n1,
            _mm256_and_si256(_mm256_cmpgt_epi8(v1, _mm256_set1_epi8(8)),
                             _mm256_cmpgt_epi8(_mm256_set1_epi8(14), v1))));
    m->hi = mm_pair(_mm256_cmpgt_epi8(_mm256_setzero_si256(), v0),
                    _mm256_cmpgt_epi8(_mm256_setzero_si256(), v1));
    m->ap = mm_pair(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('\'')),
                    _mm256_cmpeq_epi8(v1, _mm256_set1_epi8('\'')));
    m->sl = mm_pair(_mm256_cmpeq_epi8(v0, _mm256_set1_epi8('/')),
                    _mm256_cmpeq_epi8(v1, _mm256_set1_epi8('/')));
}

static const uint8_t tsimd_nemo_cls[256] = {
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x28, 0x28, 0x28, 0x28, 0x28, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x24, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    0x40, 0x40, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

typedef struct {
    uint8_t cls;
    uint8_t c2_os;
    uint8_t flags;
} ocar_t;

static inline int is_ascii_ws(uint8_t b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == 0x0b ||
           b == 0x0c;
}
static inline int is_tail_byte(uint8_t b) {
    return b == '\r' || b == '\n' || b == '/';
}
static inline int is_letter_ascii(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
}
static inline int is_other_ascii(uint8_t b) {
    return !is_letter_ascii(b) && !(b >= '0' && b <= '9') && !is_ascii_ws(b);
}

const uint8_t tsimd_nemo_bcls[256] = {
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x02, 0x03, 0x02, 0x02, 0x03, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x42, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x11,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
};

static int c2_os_at(const uint8_t *d, size_t idx) {
    uint8_t b2 = d[idx];
    if (b2 >= 0x80) return -1;
    uint8_t bb = tsimd_nemo_bcls[b2];
    if (b2 != '/')
        return (bb & 1) ? 0 : ((bb & 64) || (bb & 16));

    size_t r = idx;
    unsigned steps = 0;
    while (r > 0 && is_tail_byte(d[r - 1])) {
        r--;
        if (++steps > 8) return -1;
    }
    int trigger = INT_MAX;
    int seen_slash = 0;
    for (size_t j = 0; r + j <= idx; j++) {
        uint8_t tb = d[r + j];
        if (tb == '/') { seen_slash = 1; continue; }
        if (seen_slash) { trigger = (int)j; break; }
        if (j == 0) {
            if (r == 0) { trigger = 0; break; }
            uint8_t pb = d[r - 1];
            if (pb >= 0x80) return -1;
            if (is_other_ascii(pb)) { trigger = 0; break; }
        }
    }
    int absorbed = (int)(idx - r) >= trigger;
    return !absorbed;
}

typedef struct {
    uint64_t l, u, cl, n, o, ws;
    uint64_t w2, w3;
    uint64_t lead2, lead3, lead4;
    uint64_t cont;
    uint64_t resid;
    uint64_t mk;
} ouni_t;

static void classify_uni_nemo(const uint8_t *d, size_t scan, uint64_t m,
                              ouni_t *u) {

    while (m) {
        unsigned i = (unsigned)__builtin_ctzll(m);
        m &= m - 1;
        uint8_t b = d[scan + i];
        uint64_t lead = 1ull << i;
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t adv = tsimd_utf8_next(d + scan + i, 8, &cp, &cls);
        unsigned l = (adv == 0 || b < 0xC2) ? 1u
                                            : (adv > 4 ? 4u : (unsigned)adv);
        uint64_t chm = ((1ull << l) - 1) << i;

        if (b < 0xC2 || adv <= 1 ||
            cls < TSIMD_CLASS_OTHER || cls > TSIMD_CLASS_SPACE) {

            u->resid |= 1ull << i;
            if (b < 0xC2) continue;
            size_t e = i + l;
            while (e < 64) {
                uint8_t c = d[scan + e];
                if (c < 0x80) {
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                        c == '\v' || c == '\f' ||
                        (c >= '0' && c <= '9') ||
                        (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
                        break;
                    e++;
                    continue;
                }
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a2 = tsimd_utf8_next(d + scan + e, 8, &cp2, &cls2);
                if (a2 == 0 || cls2 != TSIMD_CLASS_OTHER) break;
                e += a2 > 4 ? 4 : (unsigned)a2;
            }
            while (e < 64 && (d[scan + e] == '\r' || d[scan + e] == '\n' ||
                              d[scan + e] == '/'))
                e++;
            if (e > i + 1) u->resid |= (((1ull << (e - i)) - 1) << i);
            m &= ~chm;
            continue;
        }
        uint8_t cf = tsimd_class_case(cp);
        if (cf == 1) {
            u->l |= chm;
            u->u |= chm;
        } else if (cf == 2) {
            u->l |= chm;
        } else if (cf == 3) {
            u->l |= chm;
            u->cl |= chm;
        } else if (cf == 4) {
            u->o |= chm;
            u->mk |= chm;
        } else if (cls == TSIMD_CLASS_NUMBER) {
            u->n |= chm;
            u->resid |= chm;
        } else if (cls == TSIMD_CLASS_SPACE) {
            u->ws |= chm;
            if (i + l > 64 || l == 4) {
                u->resid |= chm;
            } else if (l == 2) {
                u->w2 |= lead;
            } else {
                u->w3 |= lead;
            }
        } else {
            u->o |= chm;
        }
        if (l == 2) u->lead2 |= lead;
        else if (l == 3) u->lead3 |= lead;
        else if (l == 4) u->lead4 |= lead;
        u->cont |= chm & ~lead;
        m &= ~chm;
    }
}

static void nemo_carries(const uint8_t *d, size_t w, ocar_t *c) {
    c->cls = 0;
    c->c2_os = 0;
    c->flags = 0;
    if (w == 0) return;
    uint8_t b = d[w - 1];
    if (b >= 0x80) { c->flags = 2; return; }
    uint8_t bb = tsimd_nemo_bcls[b];
    if (bb & 1) {

        size_t r = w - 1;
        unsigned steps = 0;
        while (r > 0 && is_tail_byte(d[r - 1])) {
            r--;
            if (++steps > 8) { c->flags = 2; return; }
        }
        int trigger = INT_MAX;
        int seen_slash = 0;
        for (size_t j = 0; r + j < w; j++) {
            uint8_t tb = d[r + j];
            if (tb == '/') { seen_slash = 1; continue; }
            if (seen_slash) { trigger = (int)j; break; }
            if (j == 0) {
                if (r == 0) continue;
                uint8_t pb = d[r - 1];
                if (pb >= 0x80) { c->flags = 2; return; }
                if (is_other_ascii(pb)) { trigger = 0; break; }
            }
        }
        if ((int)(w - 1 - r) >= trigger) {
            c->flags = 1;
            return;
        }

        c->cls = (uint8_t)((b == '/') ? (1u << 4) : (1u << 5));
        if (w >= 2) {
            int c2 = c2_os_at(d, w - 2);
            if (c2 < 0) c->flags = 2;
            else c->c2_os = (uint8_t)c2;
        }
        return;
    }

    c->cls = tsimd_nemo_cls[b];
    if (w >= 2) {
        int c2 = c2_os_at(d, w - 2);
        if (c2 < 0) c->flags = 2;
        else c->c2_os = (uint8_t)c2;
    }
}

static void nemo_algebra_ascii(const uint8_t *d, size_t w, const amask_t *m,
                          const ocar_t *cr, const ouni_t *u,
                          uint64_t *usable_out, uint64_t *bad_out) {
    (void)u;

    const uint64_t uz_lead4 = 0;
    const uint64_t uz_lead2 = 0;
    const uint64_t uz_lead3 = 0;
    const uint64_t uz_resid = 0;
    const uint64_t uz_cont = 0;
    const uint64_t uz_w2 = 0;
    const uint64_t uz_cl = 0;
    const uint64_t uz_w3 = 0;
    const uint64_t uz_mk = 0;
    const uint64_t uz_ws = 0;
    const uint64_t uz_l = 0;
    const uint64_t uz_u = 0;
    const uint64_t uz_o = 0;

    uint64_t l = m->l, up = m->up, dg = m->d, s = m->s, wt = m->wt,
             n = m->n, ap = m->ap, sl = m->sl;

    const uint64_t cc = cr->cls;
    uint64_t pl = (cc >> 0) & 1, pu = (cc >> 1) & 1, pcl = 0,
             ps = (cc >> 2) & 1, pwt = (cc >> 3) & 1, po = (cc >> 4) & 1,
             pws = (cc >> 5) & 1, pd = (cc >> 6) & 1;
    uint64_t c2_os = cr->c2_os, b2b_in = 0;
    const int p_abs = (cr->flags >> 0) & 1;
    const int force_bad_lead = (cr->flags >> 1) & 1;

    uint64_t lb = l | uz_l;
    uint64_t ub = up | uz_u;
    uint64_t clb = uz_cl;
    uint64_t sb = s;
    uint64_t wtb = wt | uz_ws;

    uint64_t ob = ~(l | dg | s | wt | n | m->hi) | uz_o;
    uint64_t ws_all = s | wt | n | uz_ws;
    uint64_t contm = uz_cont;

    uint64_t tcls = n | sl;
    uint64_t abs_seed = (n & ((ob << 1) | po)) | (p_abs ? (tcls & 1) : 0);
    uint64_t abs_t = abs_seed == 0 ? 0 : smear_up(abs_seed, tcls);
    uint64_t ob_eff = ob & ~abs_t;
    uint64_t tail_seed = n & (((ob_eff | uz_resid) << 1) | po);
    uint64_t tmask = tail_seed ? smear_up(tail_seed, n | sl) : 0;
    ob_eff &= ~tmask;

    uint64_t len1 = ~(contm | uz_lead2 | uz_lead3 | uz_lead4);
    uint64_t c_test = ((ob_eff | sb) << 1) | po | ps;
    uint64_t b2back = ((c_test & len1) << 1) |
                      ((c_test & uz_lead2) << 2) |
                      ((c_test & uz_lead3) << 3) |
                      ((c_test & uz_lead4) << 4) | c2_os | b2b_in;
    uint64_t p_l = (lb << 1) | pl;
    uint64_t p_u = (ub << 1) | pu;
    uint64_t p_cl = (clb << 1) | pcl;
    uint64_t p_s = (sb << 1) | ps;
    uint64_t p_wt = (wtb << 1) | pwt;
    uint64_t p_o = (ob_eff << 1) | po;
    uint64_t absorb = p_o & ~b2back;
    uint64_t p_sl = p_l & ~p_u & ~p_cl;
    uint64_t b_su = ub & ~contm & p_sl;
    uint64_t b_letters = (lb & ~contm & ~p_l & ~p_s & ~p_wt & ~absorb) | b_su;

    uint64_t b_digits = dg;

    uint64_t b_punct = ob_eff & ~p_o & ~p_s;

    uint64_t bad = 0;
    if (force_bad_lead) bad |= (smear_up(tcls & 1, tcls) << 1) | 0b11;
    uint64_t resid = uz_resid;

    bad |= resid | resid << 1 | resid << 2 | resid >> 1;
    uint64_t mk = uz_mk;
    if (mk) bad |= mk | mk << 1 | mk << 2 | mk << 3 | mk << 4 | mk >> 1;
    bad |= ub & ~contm & ((clb << 1) | pcl);

    uint64_t ws_eff = ws_all & ~abs_t;
    uint8_t nb64 = d[w + 64];
    int nn64 = nb64 < 0x80 ? !is_ascii_ws(nb64) : 1;
    if (nb64 >= 0x80) {

        bad |= 0xE000000000000000ull;
        nn64 = 1;
    }

    uint64_t nn64m = nn64 ? UINT64_MAX : 0;

    uint64_t nonws = ~ws_eff;
    if ((ws_eff >> 63) != 0 && !nn64) {
        if (nonws == 0) {
            *usable_out = 0;
            *bad_out = UINT64_MAX;
            return;
        }
        unsigned hh = 63 - (unsigned)__builtin_clzll(nonws);
        bad |= UINT64_MAX << (hh + 1);
    }

    uint64_t seed = (dg & (bad << 1)) | (dg & pd);
    if (seed) bad |= smear_up(seed, dg);

    uint64_t ws_leads1 = (s | wt | n) & ws_eff;
    uint64_t ws_leads = (ws_leads1 | uz_w2 | uz_w3) & ~abs_t;
    uint64_t p_ws = (ws_eff << 1) | pws;
    uint64_t edge_last = (ws_leads1 & (1ull << 63)) |
                         (uz_w2 & (1ull << 62)) | (uz_w3 & (1ull << 61));
    uint64_t split_ok = (ws_leads1 & (nonws >> 1)) |
                        (uz_w2 & (nonws >> 2)) | (uz_w3 & (nonws >> 3)) |
                        (edge_last & nn64m);
    uint64_t b_ws = ws_leads & (~p_ws | split_ok) & ~tmask;

    uint64_t runs_n = n & ws_eff & ~bad;
    while (runs_n) {
        unsigned f = (unsigned)__builtin_ctzll(runs_n);
        uint64_t below_gap = nonws & ((1ull << f) - 1);
        unsigned a = below_gap ? 64 - (unsigned)__builtin_clzll(below_gap)
                               : 0;
        uint64_t above = nonws & (UINT64_MAX << f);
        unsigned e = above ? (unsigned)__builtin_ctzll(above) : 64;
        uint64_t run_mask = (UINT64_MAX << a) &
                            (e < 64 ? ~(UINT64_MAX << e) : UINT64_MAX);
        b_ws &= ~run_mask;
        b_ws |= 1ull << a;
        unsigned q = 63 - (unsigned)__builtin_clzll(n & run_mask);
        if ((q + 1) < e) {
            b_ws |= 1ull << (q + 1);
            uint64_t tail = run_mask & (UINT64_MAX << (q + 1));
            uint64_t tail_leads = ws_leads & tail;
            if (tail_leads)
                b_ws |= 1ull << (63 - (unsigned)__builtin_clzll(tail_leads));
        }
        runs_n &= ~run_mask;
    }

    uint64_t boundary = b_letters | b_digits | b_punct | b_ws;

    *usable_out = boundary & ~bad;
    *bad_out = bad;
}

static void nemo_algebra(const uint8_t *d, size_t w, const amask_t *m,
                          const ocar_t *cr, const ouni_t *u,
                          uint64_t *usable_out, uint64_t *bad_out) {

    const uint64_t uz_lead4 = u ? u->lead4 : 0;
    const uint64_t uz_lead2 = u ? u->lead2 : 0;
    const uint64_t uz_lead3 = u ? u->lead3 : 0;
    const uint64_t uz_resid = u ? u->resid : 0;
    const uint64_t uz_cont = u ? u->cont : 0;
    const uint64_t uz_w2 = u ? u->w2 : 0;
    const uint64_t uz_cl = u ? u->cl : 0;
    const uint64_t uz_w3 = u ? u->w3 : 0;
    const uint64_t uz_mk = u ? u->mk : 0;
    const uint64_t uz_ws = u ? u->ws : 0;
    const uint64_t uz_l = u ? u->l : 0;
    const uint64_t uz_u = u ? u->u : 0;
    const uint64_t uz_o = u ? u->o : 0;

    uint64_t l = m->l, up = m->up, dg = m->d, s = m->s, wt = m->wt,
             n = m->n, ap = m->ap, sl = m->sl;

    const uint64_t cc = cr->cls;
    uint64_t pl = (cc >> 0) & 1, pu = (cc >> 1) & 1, pcl = 0,
             ps = (cc >> 2) & 1, pwt = (cc >> 3) & 1, po = (cc >> 4) & 1,
             pws = (cc >> 5) & 1, pd = (cc >> 6) & 1;
    uint64_t c2_os = cr->c2_os, b2b_in = 0;
    const int p_abs = (cr->flags >> 0) & 1;
    const int force_bad_lead = (cr->flags >> 1) & 1;

    uint64_t lb = l | uz_l;
    uint64_t ub = up | uz_u;
    uint64_t clb = uz_cl;
    uint64_t sb = s;
    uint64_t wtb = wt | uz_ws;

    uint64_t ob = ~(l | dg | s | wt | n | m->hi) | uz_o;
    uint64_t ws_all = s | wt | n | uz_ws;
    uint64_t contm = uz_cont;

    uint64_t tcls = n | sl;
    uint64_t abs_seed = (n & ((ob << 1) | po)) | (p_abs ? (tcls & 1) : 0);
    uint64_t abs_t = abs_seed == 0 ? 0 : smear_up(abs_seed, tcls);
    uint64_t ob_eff = ob & ~abs_t;
    uint64_t tail_seed = n & (((ob_eff | uz_resid) << 1) | po);
    uint64_t tmask = tail_seed ? smear_up(tail_seed, n | sl) : 0;
    ob_eff &= ~tmask;

    uint64_t len1 = ~(contm | uz_lead2 | uz_lead3 | uz_lead4);
    uint64_t c_test = ((ob_eff | sb) << 1) | po | ps;
    uint64_t b2back = ((c_test & len1) << 1) |
                      ((c_test & uz_lead2) << 2) |
                      ((c_test & uz_lead3) << 3) |
                      ((c_test & uz_lead4) << 4) | c2_os | b2b_in;
    uint64_t p_l = (lb << 1) | pl;
    uint64_t p_u = (ub << 1) | pu;
    uint64_t p_cl = (clb << 1) | pcl;
    uint64_t p_s = (sb << 1) | ps;
    uint64_t p_wt = (wtb << 1) | pwt;
    uint64_t p_o = (ob_eff << 1) | po;
    uint64_t absorb = p_o & ~b2back;
    uint64_t p_sl = p_l & ~p_u & ~p_cl;
    uint64_t b_su = ub & ~contm & p_sl;
    uint64_t b_letters = (lb & ~contm & ~p_l & ~p_s & ~p_wt & ~absorb) | b_su;

    uint64_t b_digits = dg;

    uint64_t b_punct = ob_eff & ~p_o & ~p_s;

    uint64_t bad = 0;
    if (force_bad_lead) bad |= (smear_up(tcls & 1, tcls) << 1) | 0b11;
    uint64_t resid = uz_resid;

    bad |= resid | resid << 1 | resid << 2 | resid >> 1;
    uint64_t mk = uz_mk;
    if (mk) bad |= mk | mk << 1 | mk << 2 | mk << 3 | mk << 4 | mk >> 1;
    bad |= ub & ~contm & ((clb << 1) | pcl);

    uint64_t ws_eff = ws_all & ~abs_t;
    uint8_t nb64 = d[w + 64];
    int nn64 = nb64 < 0x80 ? !is_ascii_ws(nb64) : 1;
    if (nb64 >= 0x80) {

        bad |= 0xE000000000000000ull;
        nn64 = 1;
    }

    uint64_t nn64m = nn64 ? UINT64_MAX : 0;

    uint64_t nonws = ~ws_eff;
    if ((ws_eff >> 63) != 0 && !nn64) {
        if (nonws == 0) {
            *usable_out = 0;
            *bad_out = UINT64_MAX;
            return;
        }
        unsigned hh = 63 - (unsigned)__builtin_clzll(nonws);
        bad |= UINT64_MAX << (hh + 1);
    }

    uint64_t seed = (dg & (bad << 1)) | (dg & pd);
    if (seed) bad |= smear_up(seed, dg);

    uint64_t ws_leads1 = (s | wt | n) & ws_eff;
    uint64_t ws_leads = (ws_leads1 | uz_w2 | uz_w3) & ~abs_t;
    uint64_t p_ws = (ws_eff << 1) | pws;
    uint64_t edge_last = (ws_leads1 & (1ull << 63)) |
                         (uz_w2 & (1ull << 62)) | (uz_w3 & (1ull << 61));
    uint64_t split_ok = (ws_leads1 & (nonws >> 1)) |
                        (uz_w2 & (nonws >> 2)) | (uz_w3 & (nonws >> 3)) |
                        (edge_last & nn64m);
    uint64_t b_ws = ws_leads & (~p_ws | split_ok) & ~tmask;

    uint64_t runs_n = n & ws_eff & ~bad;
    while (runs_n) {
        unsigned f = (unsigned)__builtin_ctzll(runs_n);
        uint64_t below_gap = nonws & ((1ull << f) - 1);
        unsigned a = below_gap ? 64 - (unsigned)__builtin_clzll(below_gap)
                               : 0;
        uint64_t above = nonws & (UINT64_MAX << f);
        unsigned e = above ? (unsigned)__builtin_ctzll(above) : 64;
        uint64_t run_mask = (UINT64_MAX << a) &
                            (e < 64 ? ~(UINT64_MAX << e) : UINT64_MAX);
        b_ws &= ~run_mask;
        b_ws |= 1ull << a;
        unsigned q = 63 - (unsigned)__builtin_clzll(n & run_mask);
        if ((q + 1) < e) {
            b_ws |= 1ull << (q + 1);
            uint64_t tail = run_mask & (UINT64_MAX << (q + 1));
            uint64_t tail_leads = ws_leads & tail;
            if (tail_leads)
                b_ws |= 1ull << (63 - (unsigned)__builtin_clzll(tail_leads));
        }
        runs_n &= ~run_mask;
    }

    uint64_t boundary = b_letters | b_digits | b_punct | b_ws;

    *usable_out = boundary & ~bad;
    *bad_out = bad;
}

#define TSIMD_MASK_WIN 64

static size_t g_mask_pieces = 0, g_scalar_pieces = 0;
static size_t g_windows = 0, g_walker_wins = 0;

static int mask_stats_on(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_MASK_STATS");
        atomic_store(&state, (e && e[0] == '1') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v == 1;
}

static int no_precompute(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_NO_PRECOMPUTE");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}
void tsimd_nemo_mask_stats(size_t *mask_pieces, size_t *scalar_pieces) {

    size_t m = g_mask_pieces, s = g_scalar_pieces;
    if (mask_pieces) *mask_pieces = m;
    if (scalar_pieces) *scalar_pieces = s;
    if (mask_pieces == NULL && scalar_pieces == NULL) {
        fprintf(stderr, "windows=%zu walker_wins=%zu (%.2f%%)\n",
                g_windows, g_walker_wins,
                g_windows ? 100.0 * g_walker_wins / g_windows : 0.0);
    }
    (void)0;
}

static int bmi2_flatten_enabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_BMI2_FLATTEN");
        v = (e && e[0] == '0') ? 2
            : (tsimd_cpu_bmi2() ? 1 : 2);
        atomic_store(&state, v);
    }
    return v == 1;
}

#ifdef TSIMD_FILL_PROBE_BUILD
static void np_add(uint64_t a, uint64_t c, uint64_t uni, uint64_t alg);
#endif

static void window_usable_bad(const uint8_t *d, size_t len, size_t scan,
                              uint64_t *usable_out, uint64_t *bad_out) {
    amask_t m;
#ifdef TSIMD_FILL_PROBE_BUILD
    uint64_t np_t0 = __builtin_ia32_rdtsc();
#endif
    ascii_masks_64(d + scan, &m);
#ifdef TSIMD_FILL_PROBE_BUILD
    uint64_t np_t1 = __builtin_ia32_rdtsc();
#endif
    ocar_t cr;
    nemo_carries(d, scan, &cr);
#ifdef TSIMD_FILL_PROBE_BUILD
    uint64_t np_t2 = __builtin_ia32_rdtsc();
#endif
    ouni_t u;

    if (m.hi) {

        memset(&u, 0, sizeof(u));

        classify_uni_nemo(d, scan, m.hi, &u);
#ifdef TSIMD_FILL_PROBE_BUILD
        uint64_t np_t3 = __builtin_ia32_rdtsc();
#endif
        nemo_algebra(d, scan, &m, &cr, &u,
                     usable_out, bad_out);
#ifdef TSIMD_FILL_PROBE_BUILD
        np_add(np_t1 - np_t0, np_t2 - np_t1, np_t3 - np_t2,
               __builtin_ia32_rdtsc() - np_t3);
#endif

    } else {
#ifdef TSIMD_FILL_PROBE_BUILD
        uint64_t np_t3 = __builtin_ia32_rdtsc();
#endif
        nemo_algebra_ascii(d, scan, &m, &cr, NULL,
                     usable_out, bad_out);
#ifdef TSIMD_FILL_PROBE_BUILD
        np_add(np_t1 - np_t0, np_t2 - np_t1, 0,
               __builtin_ia32_rdtsc() - np_t3);
#endif

    }
}

static void mask_precompute(tsimd_o200k_mstate_t *st, const uint8_t *d,
                            size_t len, size_t scan) {
    if (scan + TSIMD_MASK_WIN + 8 < len) {
        window_usable_bad(d, len, scan, &st->pre_usable, &st->pre_bad);
        st->pre_scan = scan;
        st->pre_valid = 1;
    } else {
        st->pre_valid = 0;
    }
}

static const uint64_t tsimd_bitidx8[256] = {
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000001ull, 0x0000000000000100ull,
    0x0000000000000002ull, 0x0000000000000200ull, 0x0000000000000201ull, 0x0000000000020100ull,
    0x0000000000000003ull, 0x0000000000000300ull, 0x0000000000000301ull, 0x0000000000030100ull,
    0x0000000000000302ull, 0x0000000000030200ull, 0x0000000000030201ull, 0x0000000003020100ull,
    0x0000000000000004ull, 0x0000000000000400ull, 0x0000000000000401ull, 0x0000000000040100ull,
    0x0000000000000402ull, 0x0000000000040200ull, 0x0000000000040201ull, 0x0000000004020100ull,
    0x0000000000000403ull, 0x0000000000040300ull, 0x0000000000040301ull, 0x0000000004030100ull,
    0x0000000000040302ull, 0x0000000004030200ull, 0x0000000004030201ull, 0x0000000403020100ull,
    0x0000000000000005ull, 0x0000000000000500ull, 0x0000000000000501ull, 0x0000000000050100ull,
    0x0000000000000502ull, 0x0000000000050200ull, 0x0000000000050201ull, 0x0000000005020100ull,
    0x0000000000000503ull, 0x0000000000050300ull, 0x0000000000050301ull, 0x0000000005030100ull,
    0x0000000000050302ull, 0x0000000005030200ull, 0x0000000005030201ull, 0x0000000503020100ull,
    0x0000000000000504ull, 0x0000000000050400ull, 0x0000000000050401ull, 0x0000000005040100ull,
    0x0000000000050402ull, 0x0000000005040200ull, 0x0000000005040201ull, 0x0000000504020100ull,
    0x0000000000050403ull, 0x0000000005040300ull, 0x0000000005040301ull, 0x0000000504030100ull,
    0x0000000005040302ull, 0x0000000504030200ull, 0x0000000504030201ull, 0x0000050403020100ull,
    0x0000000000000006ull, 0x0000000000000600ull, 0x0000000000000601ull, 0x0000000000060100ull,
    0x0000000000000602ull, 0x0000000000060200ull, 0x0000000000060201ull, 0x0000000006020100ull,
    0x0000000000000603ull, 0x0000000000060300ull, 0x0000000000060301ull, 0x0000000006030100ull,
    0x0000000000060302ull, 0x0000000006030200ull, 0x0000000006030201ull, 0x0000000603020100ull,
    0x0000000000000604ull, 0x0000000000060400ull, 0x0000000000060401ull, 0x0000000006040100ull,
    0x0000000000060402ull, 0x0000000006040200ull, 0x0000000006040201ull, 0x0000000604020100ull,
    0x0000000000060403ull, 0x0000000006040300ull, 0x0000000006040301ull, 0x0000000604030100ull,
    0x0000000006040302ull, 0x0000000604030200ull, 0x0000000604030201ull, 0x0000060403020100ull,
    0x0000000000000605ull, 0x0000000000060500ull, 0x0000000000060501ull, 0x0000000006050100ull,
    0x0000000000060502ull, 0x0000000006050200ull, 0x0000000006050201ull, 0x0000000605020100ull,
    0x0000000000060503ull, 0x0000000006050300ull, 0x0000000006050301ull, 0x0000000605030100ull,
    0x0000000006050302ull, 0x0000000605030200ull, 0x0000000605030201ull, 0x0000060503020100ull,
    0x0000000000060504ull, 0x0000000006050400ull, 0x0000000006050401ull, 0x0000000605040100ull,
    0x0000000006050402ull, 0x0000000605040200ull, 0x0000000605040201ull, 0x0000060504020100ull,
    0x0000000006050403ull, 0x0000000605040300ull, 0x0000000605040301ull, 0x0000060504030100ull,
    0x0000000605040302ull, 0x0000060504030200ull, 0x0000060504030201ull, 0x0006050403020100ull,
    0x0000000000000007ull, 0x0000000000000700ull, 0x0000000000000701ull, 0x0000000000070100ull,
    0x0000000000000702ull, 0x0000000000070200ull, 0x0000000000070201ull, 0x0000000007020100ull,
    0x0000000000000703ull, 0x0000000000070300ull, 0x0000000000070301ull, 0x0000000007030100ull,
    0x0000000000070302ull, 0x0000000007030200ull, 0x0000000007030201ull, 0x0000000703020100ull,
    0x0000000000000704ull, 0x0000000000070400ull, 0x0000000000070401ull, 0x0000000007040100ull,
    0x0000000000070402ull, 0x0000000007040200ull, 0x0000000007040201ull, 0x0000000704020100ull,
    0x0000000000070403ull, 0x0000000007040300ull, 0x0000000007040301ull, 0x0000000704030100ull,
    0x0000000007040302ull, 0x0000000704030200ull, 0x0000000704030201ull, 0x0000070403020100ull,
    0x0000000000000705ull, 0x0000000000070500ull, 0x0000000000070501ull, 0x0000000007050100ull,
    0x0000000000070502ull, 0x0000000007050200ull, 0x0000000007050201ull, 0x0000000705020100ull,
    0x0000000000070503ull, 0x0000000007050300ull, 0x0000000007050301ull, 0x0000000705030100ull,
    0x0000000007050302ull, 0x0000000705030200ull, 0x0000000705030201ull, 0x0000070503020100ull,
    0x0000000000070504ull, 0x0000000007050400ull, 0x0000000007050401ull, 0x0000000705040100ull,
    0x0000000007050402ull, 0x0000000705040200ull, 0x0000000705040201ull, 0x0000070504020100ull,
    0x0000000007050403ull, 0x0000000705040300ull, 0x0000000705040301ull, 0x0000070504030100ull,
    0x0000000705040302ull, 0x0000070504030200ull, 0x0000070504030201ull, 0x0007050403020100ull,
    0x0000000000000706ull, 0x0000000000070600ull, 0x0000000000070601ull, 0x0000000007060100ull,
    0x0000000000070602ull, 0x0000000007060200ull, 0x0000000007060201ull, 0x0000000706020100ull,
    0x0000000000070603ull, 0x0000000007060300ull, 0x0000000007060301ull, 0x0000000706030100ull,
    0x0000000007060302ull, 0x0000000706030200ull, 0x0000000706030201ull, 0x0000070603020100ull,
    0x0000000000070604ull, 0x0000000007060400ull, 0x0000000007060401ull, 0x0000000706040100ull,
    0x0000000007060402ull, 0x0000000706040200ull, 0x0000000706040201ull, 0x0000070604020100ull,
    0x0000000007060403ull, 0x0000000706040300ull, 0x0000000706040301ull, 0x0000070604030100ull,
    0x0000000706040302ull, 0x0000070604030200ull, 0x0000070604030201ull, 0x0007060403020100ull,
    0x0000000000070605ull, 0x0000000007060500ull, 0x0000000007060501ull, 0x0000000706050100ull,
    0x0000000007060502ull, 0x0000000706050200ull, 0x0000000706050201ull, 0x0000070605020100ull,
    0x0000000007060503ull, 0x0000000706050300ull, 0x0000000706050301ull, 0x0000070605030100ull,
    0x0000000706050302ull, 0x0000070605030200ull, 0x0000070605030201ull, 0x0007060503020100ull,
    0x0000000007060504ull, 0x0000000706050400ull, 0x0000000706050401ull, 0x0000070605040100ull,
    0x0000000706050402ull, 0x0000070605040200ull, 0x0000070605040201ull, 0x0007060504020100ull,
    0x0000000706050403ull, 0x0000070605040300ull, 0x0000070605040301ull, 0x0007060504030100ull,
    0x0000070605040302ull, 0x0007060504030200ull, 0x0007060504030201ull, 0x0706050403020100ull,
};

size_t tsimd_nemo_flatten_bounds(uint64_t bits, uint32_t scan_base,
                                  uint32_t *out, size_t cap) {
    if (!bits || cap == 0) return 0;
    size_t n = 0;
    for (int i = 0; i < 8; i++, bits >>= 8) {
        unsigned v = (unsigned)(bits & 0xFF);
        if (!v) continue;
        unsigned c = __builtin_popcount(v);
        uint64_t p = tsimd_bitidx8[v];
        uint32_t base8 = scan_base + 8u * (unsigned)i;
        if (n + c <= cap) {

            switch (c) {
                case 8: out[n + 7] = base8 + (uint8_t)(p >> 56);
                case 7: out[n + 6] = base8 + (uint8_t)(p >> 48);
                case 6: out[n + 5] = base8 + (uint8_t)(p >> 40);
                case 5: out[n + 4] = base8 + (uint8_t)(p >> 32);
                case 4: out[n + 3] = base8 + (uint8_t)(p >> 24);
                case 3: out[n + 2] = base8 + (uint8_t)(p >> 16);
                case 2: out[n + 1] = base8 + (uint8_t)(p >> 8);
                case 1: out[n] = base8 + (uint8_t)p;
            }
            n += c;
        } else {

            for (unsigned j = 0; j < c && n < cap; j++, p >>= 8)
                out[n++] = base8 + (uint32_t)(p & 0xFF);
            return n;
        }
    }
    return n;
}

__attribute__((target("bmi2")))
size_t tsimd_nemo_flatten_bounds_bmi2(uint64_t bits, uint32_t scan_base,
                                       uint32_t *out, size_t cap) {
    if (!bits || cap == 0) return 0;
    size_t n = 0;
    for (int i = 0; i < 8; i++, bits >>= 8) {
        unsigned v = (unsigned)(bits & 0xFF);
        if (!v) continue;
        unsigned c = __builtin_popcount(v);

        uint64_t expand = __builtin_ia32_pdep_di(v, 0x0101010101010101ULL);
        uint64_t b0 = __builtin_ia32_pext_di(0x0100010001000100ULL, expand);
        uint64_t b1 = __builtin_ia32_pext_di(0x0101000001010000ULL, expand);
        uint64_t b2 = __builtin_ia32_pext_di(0x0101010100000000ULL, expand);
        uint64_t p = 0;
        for (unsigned j = 0; j < c; j++)
            p |= (uint64_t)(((b0 >> j) & 1) | (((b1 >> j) & 1) << 1) |
                            (((b2 >> j) & 1) << 2))
                 << (8 * j);
        uint32_t base8 = scan_base + 8u * (unsigned)i;
        if (n + c <= cap) {
            switch (c) {
                case 8: out[n + 7] = base8 + (uint8_t)(p >> 56);
                case 7: out[n + 6] = base8 + (uint8_t)(p >> 48);
                case 6: out[n + 5] = base8 + (uint8_t)(p >> 40);
                case 5: out[n + 4] = base8 + (uint8_t)(p >> 32);
                case 4: out[n + 3] = base8 + (uint8_t)(p >> 24);
                case 3: out[n + 2] = base8 + (uint8_t)(p >> 16);
                case 2: out[n + 1] = base8 + (uint8_t)(p >> 8);
                case 1: out[n] = base8 + (uint8_t)p;
            }
            n += c;
        } else {
            for (unsigned j = 0; j < c && n < cap; j++, p >>= 8)
                out[n++] = base8 + (uint32_t)(p & 0xFF);
            return n;
        }
    }
    return n;
}

static int mask_harvest(tsimd_o200k_mstate_t *st, const uint8_t *d,
                        size_t len, size_t base, uint32_t *bounds,
                        size_t cap, size_t *nb) {
    size_t pos = st->pos, scan = st->scan;
    const int st_on = mask_stats_on();

    if (pos >= scan + TSIMD_MASK_WIN) scan = pos & ~(size_t)63;
    if (scan + TSIMD_MASK_WIN + 8 >= len) {

        while (pos < len && *nb < cap) {
            size_t e = tsimd_match_nemo(d, len, pos);
            if (e <= pos || e > len) return -1;
            if (st_on) g_scalar_pieces++;
            if (*nb < cap) bounds[(*nb)++] = (uint32_t)(e - base);
            pos = e;
        }
        st->pos = pos;
        st->scan = scan;
        return 0;
    }
    size_t k = pos >= scan ? pos - scan : 0;
    int crossed = pos < scan;
    uint64_t usable, bad;
    if (st->pre_valid && st->pre_scan == scan) {
        usable = st->pre_usable;
        bad = st->pre_bad;
    } else {
        window_usable_bad(d, len, scan, &usable, &bad);
    }
    uint64_t need = crossed ? ~(uint64_t)0
                            : ~((1ull << k) - 1);
    if (st_on) g_windows++;
    if (bad & need) {
        if (st_on) g_walker_wins++;

        size_t cur = pos;
        for (;;) {
            if (cur >= scan + TSIMD_MASK_WIN || cur >= len || *nb >= cap)
                break;
            size_t kc = cur >= scan ? cur - scan : 0;
            uint64_t ab = usable & ~((1ull << kc) - 1);
            if (kc == 0 && crossed) ab = usable;
            uint64_t sb = bad & ~((1ull << kc) - 1);
            if (!sb) {
                while (ab && *nb < cap) {
                    unsigned k2 = (unsigned)__builtin_ctzll(ab);
                    size_t bp = scan + k2;
                    if (bp > cur) {
                        if (st_on) g_mask_pieces++;
                        if (*nb < cap)
                            bounds[(*nb)++] = (uint32_t)(bp - base);
                    }
                    cur = bp;
                    ab &= ab - 1;
                }
                break;
            }
            unsigned fb = (unsigned)__builtin_ctzll(sb);
            uint64_t prefix = ab & ((1ull << fb) - 1);
            while (prefix && *nb < cap) {
                unsigned k2 = (unsigned)__builtin_ctzll(prefix);
                size_t bp = scan + k2;
                if (bp > cur) {
                    if (st_on) g_mask_pieces++;
                    if (*nb < cap)
                        bounds[(*nb)++] = (uint32_t)(bp - base);
                }
                cur = bp;
                prefix &= prefix - 1;
            }
            uint64_t rest = usable & (UINT64_MAX << fb);
            size_t until = rest ? scan + (size_t)__builtin_ctzll(rest)
                                : scan + TSIMD_MASK_WIN;
            while (cur < until && cur < len && *nb < cap) {
                size_t e = tsimd_match_nemo(d, len, cur);
                if (e <= cur || e > len) return -1;
                if (st_on) g_scalar_pieces++;
                if (*nb < cap) bounds[(*nb)++] = (uint32_t)(e - base);
                cur = e;
            }
        }
        scan += TSIMD_MASK_WIN;
        st->pos = pos = cur;
        st->scan = scan;
        mask_precompute(st, d, len, scan);
        return 0;
    }
    uint64_t bits = crossed ? usable : usable & (need << 1);
    while (bits && *nb < cap) {

        unsigned k2 = (unsigned)__builtin_ctzll(bits);
        size_t bpos = scan + k2;
        if (bpos > pos) {
            if (st_on) g_mask_pieces++;
            if (*nb < cap) bounds[(*nb)++] = (uint32_t)(bpos - base);
        }
        pos = bpos;
        bits &= bits - 1;
    }
    scan += TSIMD_MASK_WIN;
    st->pos = pos;
    st->scan = scan;
    if (getenv("TSIMD_FILL_DBG") && len - pos < 16)
        fprintf(stderr, "FUSED exit pos=%zu scan=%zu len=%zu remaining=%zu\n",
                pos, scan, len, len - pos);
    mask_precompute(st, d, len, scan);
    return 0;
}

static inline void fused_put(tsimd_pcache_span_t *ents, size_t cap,
                             size_t *n, const uint8_t *p, size_t el,
                             const uint8_t *d, size_t len,
                             tsimd_pretok_cache_t *pc, int hcrc) {
    if (*n >= cap) return;
    ents[*n].ptr = p;
    if (el <= TSIMD_PCACHE_MAX_SPAN) {
        tsimd_pcache_key_inline_arm(p, el, d + len,
                                    &ents[*n].k0, &ents[*n].k1,
                                    &ents[*n].meta, hcrc);

        if (pc) tsimd_pcache_prefetch_l2(pc, ents[*n].meta);
    } else {
        ents[*n].k0 = 0;
        ents[*n].k1 = 1;

        ents[*n].meta = el;
    }
    (*n)++;
}

static int mask_emit_fused(tsimd_o200k_mstate_t *st, const uint8_t *d,
                           size_t len, tsimd_pcache_span_t *ents,
                           tsimd_pretok_cache_t *pc, size_t cap,
                           size_t *n, int hcrc) {
    size_t pos = st->pos, scan = st->scan;
    const int st_on = mask_stats_on();

    for (;;) {

        if (pos >= scan + TSIMD_MASK_WIN) scan = pos & ~(size_t)63;
        if (scan + TSIMD_MASK_WIN + 8 >= len) {
            while (pos < len && *n < cap) {
                size_t e = tsimd_match_nemo(d, len, pos);
                if (e <= pos || e > len) return -1;
                if (st_on) g_scalar_pieces++;
                fused_put(ents, cap, n, d + pos, e - pos, d, len, pc,
                          hcrc);
                pos = e;
            }
            if (getenv("TSIMD_FILL_DBG"))
                fprintf(stderr, "FUSED scaltail exit pos=%zu n=%zu len=%zu rem=%zu\n", pos, *n, len, len - pos);
            st->pos = pos;
            st->scan = scan;
            return 0;
        }
        size_t k = pos >= scan ? pos - scan : 0;
        int crossed = pos < scan;
        uint64_t usable, bad;
        if (no_precompute()) { window_usable_bad(d, len, scan, &usable, &bad); }
        else if (st->pre_valid && st->pre_scan == scan) {
            usable = st->pre_usable;
            bad = st->pre_bad;
        } else {
            window_usable_bad(d, len, scan, &usable, &bad);
        }
        uint64_t need = crossed ? ~(uint64_t)0
                                : ~((1ull << k) - 1);
        if (st_on) g_windows++;
        if (bad & need) {
            static atomic_int trace_state = 0;
            int trace;
            if (trace_state) {
                trace = trace_state == 2;
            } else {
                const char *te = getenv("TSIMD_FILL_TRACE");
                trace = te && te[0] == '1';
                atomic_store(&trace_state, trace ? 2 : 1);
            }
            if (trace)
                fprintf(stderr,
                        "WALK scan=%zu pos=%zu n=%zu cap=%zu u=%016llx "
                        "b=%016llx\n",
                        scan, pos, *n, cap, (unsigned long long)usable,
                        (unsigned long long)bad);
            if (st_on) g_walker_wins++;
            size_t cur = pos;
            for (;;) {
                if (cur >= scan + TSIMD_MASK_WIN || cur >= len ||
                    *n >= cap)
                    break;
                size_t kc = cur >= scan ? cur - scan : 0;
                uint64_t ab = usable & ~((1ull << kc) - 1);
                if (kc == 0 && crossed) ab = usable;
                uint64_t sb = bad & ~((1ull << kc) - 1);
                if (!sb) {
                    while (ab && *n < cap) {
                        unsigned k2 = (unsigned)__builtin_ctzll(ab);
                        size_t bp = scan + k2;
                        if (bp > cur) {
                            if (st_on) g_mask_pieces++;
                            fused_put(ents, cap, n, d + cur, bp - cur, d,
                                      len, pc, hcrc);
                        }
                        cur = bp;
                        ab &= ab - 1;
                    }
                    break;
                }
                unsigned fb = (unsigned)__builtin_ctzll(sb);
                uint64_t prefix = ab & ((1ull << fb) - 1);
                while (prefix && *n < cap) {
                    unsigned k2 = (unsigned)__builtin_ctzll(prefix);
                    size_t bp = scan + k2;
                    if (bp > cur) {
                        if (st_on) g_mask_pieces++;
                        fused_put(ents, cap, n, d + cur, bp - cur, d, len,
                                  pc, hcrc);
                    }
                    cur = bp;
                    prefix &= prefix - 1;
                }
                uint64_t rest = usable & (UINT64_MAX << fb);
                size_t until = rest ? scan + (size_t)__builtin_ctzll(rest)
                                    : scan + TSIMD_MASK_WIN;
                while (cur < until && cur < len && *n < cap) {
                    size_t e = tsimd_match_nemo(d, len, cur);
                    if (e <= cur || e > len) return -1;
                    if (st_on) g_scalar_pieces++;
                    fused_put(ents, cap, n, d + cur, e - cur, d, len, pc,
                          hcrc);
                    cur = e;
                }
            }
            pos = cur;
        } else {
            uint64_t bits = crossed ? usable : usable & (need << 1);
            while (bits && *n < cap) {
                unsigned k2 = (unsigned)__builtin_ctzll(bits);
                size_t bpos = scan + k2;
                if (bpos > pos) {
                    if (st_on) g_mask_pieces++;
                    fused_put(ents, cap, n, d + pos, bpos - pos, d, len,
                              pc, hcrc);
                }
                pos = bpos;
                bits &= bits - 1;
            }
        }
        scan += TSIMD_MASK_WIN;
        if (*n + TSIMD_MASK_WIN > cap || pos >= len ||
            scan + TSIMD_MASK_WIN + 8 >= len)
            break;
    }
    if (getenv("TSIMD_FILL_DBG") && len - pos < 32)
        fprintf(stderr, "FUSED batchcap exit pos=%zu n=%zu len=%zu rem=%zu\n", pos, *n, len, len - pos);
    st->pos = pos;
    st->scan = scan;
    mask_precompute(st, d, len, scan);
    return 0;
}

static void emit_keyed_spans(tsimd_pcache_span_t *ents, size_t cap,
                             tsimd_pretok_cache_t *pc, const uint8_t *d,
                             size_t len, size_t base, size_t nb,
                             const uint32_t *bounds, size_t *n) {
    size_t prev = 0;
    if (nb != 0 && base + bounds[nb - 1] + 16 <= len) {
        for (size_t i = 0; i < nb; i++) {
            if (*n >= cap) return;
            size_t end = bounds[i];
            size_t el = end - prev;
            const uint8_t *p = d + base + prev;
            prev = end;
            unsigned m = el < 15 ? (unsigned)el : 15;
            uint64_t keep = (el <= TSIMD_PCACHE_MAX_SPAN) ? ~0ull : 0;
            uint64_t a, b;
            memcpy(&a, p, 8);
            memcpy(&b, p + 8, 8);
            a = (a & tsimd_pack_masks[m][0]) & keep;
            b = ((b & tsimd_pack_masks[m][1]) |
                 ((uint64_t)m << 56)) & keep;
            ents[*n].ptr = p;
            ents[*n].k0 = a;
            ents[*n].k1 = b;
            ents[*n].meta = (tsimd_pcache_hash_words_64(a, b) & keep) |
                            ((uint64_t)el & ~keep);

            if (a == 0 && keep != 0) {
                ents[*n].k1 = 0;
                ents[*n].meta = (uint64_t)el;
            } else if (pc && keep != 0) {
                tsimd_pcache_prefetch_l2(pc, ents[*n].meta);
            }
            (*n)++;
        }
    } else {
        for (size_t i = 0; i < nb; i++) {
            if (*n >= cap) return;
            size_t end = bounds[i];
            size_t el = end - prev;
            const uint8_t *p = d + base + prev;
            prev = end;
            ents[*n].ptr = p;
            if (el <= TSIMD_PCACHE_MAX_SPAN) {
                tsimd_pcache_key_inline(p, el, d + len,
                                        &ents[*n].k0, &ents[*n].k1,
                                        &ents[*n].meta);

                if (ents[*n].k0 == 0) {
                    ents[*n].k1 = 0;
                    ents[*n].meta = el;
                } else if (pc) {
                    tsimd_pcache_prefetch_l2(pc, ents[*n].meta);
                }
            } else {
                ents[*n].k0 = 0;
                ents[*n].k1 = 0;
                ents[*n].meta = el;
            }
            (*n)++;
        }
    }
}

static void emit_plain_spans(size_t *starts, size_t *ends, size_t cap,
                             const uint8_t *d, size_t base, size_t nb,
                             const uint32_t *bounds, size_t *n) {
    size_t prev = 0;
    for (size_t i = 0; i < nb; i++) {
        if (*n >= cap) return;
        size_t end = bounds[i];
        starts[*n] = base + prev;
        ends[*n] = base + end;
        prev = end;
        (*n)++;
    }
}

int tsimd_nemo_mask_fill(const uint8_t *d, size_t len, void *state,
                          size_t *starts, size_t *ends, size_t cap) {
    tsimd_o200k_mstate_t *st = state;
    size_t n = 0;
    uint32_t bounds[TSIMD_MASK_WIN + 256];
    while (n + TSIMD_MASK_WIN <= cap && st->pos < len) {
        size_t base = st->pos;
        size_t nb = 0;
        if (mask_harvest(st, d, len, base, bounds, cap - n, &nb) != 0)
            return -1;
        emit_plain_spans(starts, ends, cap, d, base, nb, bounds, &n);
        if (st->pos >= len) break;
    }
    return (int)n;
}

int tsimd_nemo_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                                tsimd_pcache_span_t *ents, size_t cap,
                                tsimd_pretok_cache_t *pc, int twophase) {
    tsimd_o200k_mstate_t *st = state;
    size_t n = 0;
    if (!twophase) {

        const int hcrc = tsimd_hash_use_crc32();

        while (n + TSIMD_MASK_WIN <= cap && st->pos < len) {
            if (mask_emit_fused(st, d, len, ents, pc, cap, &n, hcrc) != 0)
                return -1;
            if (st->pos >= len) break;
        }
        return (int)n;
    }

    uint32_t bounds[TSIMD_MASK_WIN + 256];
    while (n + TSIMD_MASK_WIN <= cap && st->pos < len) {
        size_t base = st->pos;
        size_t nb = 0;
        if (mask_harvest(st, d, len, base, bounds, cap - n, &nb) != 0)
            return -1;
        emit_keyed_spans(ents, cap, pc, d, len, base, nb, bounds, &n);
        if (st->pos >= len) break;
    }
    return (int)n;
}

int tsimd_pretokenize_nemo_masks(const uint8_t *data, size_t len,
                                  tsimd_span_array_t *out) {
    tsimd_o200k_mstate_t st = {0, 0};
    size_t starts[TSIMD_MASK_WIN + 64], ends[TSIMD_MASK_WIN + 64];
    for (;;) {
        int n = tsimd_nemo_mask_fill(data, len, &st, starts, ends,
                                      TSIMD_MASK_WIN + 64);
        if (n < 0) return -1;
        for (int i = 0; i < n; i++)
            if (tsimd_span_array_push(out, (uint32_t)starts[i],
                                      (uint32_t)ends[i]) != 0)
                return -1;
        if (n == 0 || st.pos >= len) break;
    }
    return 0;
}

#ifdef TSIMD_FILL_PROBE_BUILD

static _Atomic uint64_t np_ascii, np_carry, np_uni, np_alg, np_wins;

static void np_add(uint64_t a, uint64_t c, uint64_t uni, uint64_t alg) {
    uint64_t w = atomic_fetch_add(&np_wins, 1) + 1;
    atomic_fetch_add(&np_ascii, a);
    atomic_fetch_add(&np_carry, c);
    atomic_fetch_add(&np_uni, uni);
    atomic_fetch_add(&np_alg, alg);
    if ((w & ((1u << 20) - 1)) == 0) {
        double wd = (double)w;
        fprintf(stderr,
                "nemo fill probe: per window - ascii %.1f + carries %.1f + "
                "uni %.1f (classify %.1f), algebra %.1f, sum %.1f, %llu windows, "
                "mask %zu scalar %zu\n",
                (double)atomic_load(&np_ascii) / wd,
                (double)atomic_load(&np_carry) / wd,
                (double)atomic_load(&np_uni) / wd,
                (double)(atomic_load(&np_ascii) + atomic_load(&np_carry) +
                         atomic_load(&np_uni)) / wd,
                (double)atomic_load(&np_alg) / wd,
                (double)(atomic_load(&np_ascii) + atomic_load(&np_carry) +
                         atomic_load(&np_uni) + atomic_load(&np_alg)) / wd,
                (unsigned long long)w,
                g_mask_pieces, g_scalar_pieces);
    }
}
#endif
