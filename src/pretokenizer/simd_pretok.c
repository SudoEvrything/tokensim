#include "internal.h"
#include "simd.h"
#include "simd_internal.h"
#include "unicode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TSIMD_BLOCK 64

typedef struct {
    const uint8_t *data;
    size_t len;
    int is_cl100k;
    int is_o200k;
    int ci;
    tsimd_classify_fn_t classify;
    tsimd_span_array_t *out;
    tsimd_span_fn fn;
    void *ud;
    int err;

    size_t span_start;
    size_t base;

    uint8_t hb[4];
    uint8_t hc[4];

    uint8_t hcase[4];

    int d_open;
    unsigned d_phase;
    int d1, d2;

    int op_carry;

    int cache_valid;
    size_t cache_base;
    tsimd_cls_block_t cache;
} sw_t;

static uint8_t sw_byte(const sw_t *st, long i) {
    if (i >= 0) return st->data[st->base + (size_t)i];
    return st->hb[-i - 1];
}

static uint8_t sw_cls(const sw_t *st, long i) {
    if (i >= 0) return tsimd_class_byte(st->data[st->base + (size_t)i]);
    return st->hc[-i - 1];
}

static int sw_is_ws(const sw_t *st, long k) {
    if (k < 0) return 0;
    uint8_t b = st->data[st->base + (size_t)k];
    if (b < 0x80) {
        uint8_t c = tsimd_class_byte(b);
        return c >= CLS_SP && c <= CLS_C;
    }
    uint32_t cp;
    tsimd_char_class_t xc;
    size_t adv = tsimd_utf8_next(st->data + st->base + (size_t)k,
                                 st->len - st->base - (size_t)k, &cp, &xc);
    return adv > 0 && xc == TSIMD_CLASS_SPACE;
}

static int in_ldwc(uint8_t c) {
    return c == CLS_L || c == CLS_D || c == CLS_W || c == CLS_C;
}

static uint8_t sw_case(const sw_t *st, long i) {
    if (i >= 0) {
        uint8_t b = st->data[st->base + (size_t)i];
        if (b >= 'A' && b <= 'Z') return 1;
        if (b >= 'a' && b <= 'z') return 2;
        return 0;
    }
    if (i >= -4) return st->hcase[-i - 1];
    return 0;
}

static int suf1(uint8_t b, int ci) {
    if (ci) b = (uint8_t)(b | 0x20);
    return b == 's' || b == 't' || b == 'm' || b == 'd';
}

static int pair23(uint8_t a, uint8_t b, int ci) {
    if (ci) {
        a = (uint8_t)(a | 0x20);
        b = (uint8_t)(b | 0x20);
    }
    return (a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
           (a == 'l' && b == 'l');
}

static int o200k_bnd(const sw_t *st, long i) {
    uint8_t c1 = sw_cls(st, i - 1), c0 = sw_cls(st, i);
    int ws1 = (c1 >= CLS_SP && c1 <= CLS_C);
    int ws0 = (c0 >= CLS_SP && c0 <= CLS_C);

    int cc = (c0 != c1);
    int supp = 0;
    supp |= ws0 && ws1;
    supp |= (c1 == CLS_SP) && (c0 == CLS_L || c0 == CLS_O);
    supp |= (c1 == CLS_W) && (c0 == CLS_L);
    supp |= (c0 == CLS_C) && (c1 == CLS_O);
    supp |= (c1 == CLS_O) && (c0 == CLS_L) &&
            !((sw_cls(st, i - 2) == CLS_O) ||
              (sw_cls(st, i - 2) == CLS_SP));

    if (c0 == CLS_O && c1 == CLS_C && sw_byte(st, i) == '/') {
        long k = i - 1;
        while (k >= -4 && (sw_cls(st, k) == CLS_C ||
                           (sw_cls(st, k) == CLS_O &&
                            sw_byte(st, k) == '/')))
            k--;
        int opref = (k >= -4) ? sw_cls(st, k) == CLS_O : st->op_carry;
        if (opref) supp = 1;
    }

    uint8_t case1 = sw_case(st, i - 1), case0 = sw_case(st, i);

    int caseB = (c1 == CLS_L) && (c0 == CLS_L) &&
                case1 == 2 && case0 == 1;

    int forceD = 0;
    if (c0 == CLS_D && c1 == CLS_D) {
        long k = i;
        while (k > 0 && sw_cls(st, k - 1) == CLS_D) k--;
        if (k == 0 && st->d_open)
            forceD = ((i + (long)st->d_phase) % 3 == 0);
        else
            forceD = ((i - k) % 3 == 0);
    }
    int forceA = 0;
    if (ws0 && c1 == CLS_C) {
        long k = i - 1;
        while (k >= -4 && (sw_cls(st, k) == CLS_C ||
                           (st->is_o200k && sw_cls(st, k) == CLS_O &&
                            sw_byte(st, k) == '/')))
            k--;
        if (k >= -4 ? sw_cls(st, k) == CLS_O : st->op_carry)
            forceA = (c0 == CLS_SP || c0 == CLS_W);
    }
    return (cc && !supp) || caseB || forceD || forceA;
}

static int gpt2_bnd(const sw_t *st, long i) {
    uint8_t c1 = sw_cls(st, i - 1), c0 = sw_cls(st, i);
    uint8_t f1 = (c1 >= CLS_SP && c1 <= CLS_C) ? 4 : c1;
    uint8_t f0 = (c0 >= CLS_SP && c0 <= CLS_C) ? 4 : c0;

    int cc = (f0 != f1);
    int suppA = (sw_byte(st, i - 1) == 0x20) && (f0 != 4);
    int suppC = (sw_byte(st, i - 1) == '\'') &&
                in_ldwc(sw_cls(st, i - 2)) &&
                (suf1(sw_byte(st, i), 0) ||
                 ((long)st->base + i + 1 < (long)st->len &&
                  pair23(sw_byte(st, i), sw_byte(st, i + 1), 0)));
    int forceC =
        ((sw_byte(st, i - 2) == '\'') && suf1(sw_byte(st, i - 1), 0) &&
         in_ldwc(sw_cls(st, i - 3))) ||
        ((sw_byte(st, i - 3) == '\'') &&
         pair23(sw_byte(st, i - 2), sw_byte(st, i - 1), 0) &&
         in_ldwc(sw_cls(st, i - 4)));
    int forceB = (f1 == 4) && (f0 == 4) &&
                 (long)st->base + i + 1 < (long)st->len &&
                 !sw_is_ws(st, i + 1);
    return (cc && !suppA && !suppC) || forceB || forceC;
}

static int cl100k_bnd(const sw_t *st, long i) {
    uint8_t c1 = sw_cls(st, i - 1), c0 = sw_cls(st, i);
    int ws1 = (c1 >= CLS_SP && c1 <= CLS_C);
    int ws0 = (c0 >= CLS_SP && c0 <= CLS_C);

    int cc = (c0 != c1);
    int supp = 0;
    supp |= ws0 && ws1;
    supp |= (c1 == CLS_SP) && (c0 == CLS_L || c0 == CLS_O);
    supp |= (c1 == CLS_W) && (c0 == CLS_L);
    supp |= (c0 == CLS_C) && (c1 == CLS_O);
    supp |= (c1 == CLS_O) && (c0 == CLS_L) &&
            !((sw_cls(st, i - 2) == CLS_O) ||
              (sw_cls(st, i - 2) == CLS_SP));
    int suppC = (sw_byte(st, i - 1) == '\'') &&
                in_ldwc(sw_cls(st, i - 2)) &&
                (suf1(sw_byte(st, i), 1) ||
                 ((long)st->base + i + 1 < (long)st->len &&
                  pair23(sw_byte(st, i), sw_byte(st, i + 1), 1)));
    int forceC =
        ((sw_byte(st, i - 2) == '\'') && suf1(sw_byte(st, i - 1), 1) &&
         in_ldwc(sw_cls(st, i - 3))) ||
        ((sw_byte(st, i - 3) == '\'') &&
         pair23(sw_byte(st, i - 2), sw_byte(st, i - 1), 1) &&
         in_ldwc(sw_cls(st, i - 4)));
    int forceD = 0;
    if (c0 == CLS_D && c1 == CLS_D) {
        long k = i;
        while (k > 0 && sw_cls(st, k - 1) == CLS_D) k--;
        if (k == 0 && st->d_open)
            forceD = ((i + (long)st->d_phase) % 3 == 0);
        else
            forceD = ((i - k) % 3 == 0);
    }
    int forceA = 0;
    if (ws0 && c1 == CLS_C) {

        long k = i - 1;
        while (k >= -4 && sw_cls(st, k) == CLS_C) k--;
        if (k >= -4 ? sw_cls(st, k) == CLS_O : st->op_carry)
            forceA = (c0 == CLS_SP || c0 == CLS_W);
    }
    return (cc && !supp && !suppC) || forceC || forceD || forceA;
}

static uint64_t gpt2_mask(const sw_t *st, const tsimd_cls_block_t *b,
                          int next_is_s) {
    uint64_t S = b->sp | b->w | b->c;
    uint64_t LDWC = b->l | b->d | b->w | b->c;

    uint8_t p1 = st->hc[0];
    uint8_t f1 = (p1 >= CLS_SP && p1 <= CLS_C) ? 4 : p1;
    uint64_t cL = (f1 == CLS_L), cD = (f1 == CLS_D), cS = (f1 == 4),
             cO = (f1 == CLS_O);

    uint64_t eq = (b->l & ((b->l << 1) | cL)) |
                  (b->d & ((b->d << 1) | cD)) |
                  (S & ((S << 1) | cS)) |
                  (b->o & ((b->o << 1) | cO));
    uint64_t cc = ~eq;

    uint64_t suppA = ((b->sp << 1) | (uint64_t)(p1 == CLS_SP)) & ~S;
    uint64_t P = (b->r & (b->e >> 1)) | (b->v & (b->e >> 1)) |
                 (b->l1 & (b->l1 >> 1));
    uint64_t suppC = ((b->ap << 1) | (uint64_t)(st->hb[0] == '\'')) &
                     ((LDWC << 2) | (uint64_t)in_ldwc(st->hc[1])) &
                     (b->suf | P);
    uint64_t forceC = (((b->ap << 2) | (uint64_t)(st->hb[1] == '\'')) &
                       ((b->suf << 1) | (uint64_t)suf1(st->hb[0], 0)) &
                       ((LDWC << 3) | (uint64_t)in_ldwc(st->hc[2]))) |
                      (((b->ap << 3) | (uint64_t)(st->hb[2] == '\'')) &
                       ((P << 2) | (uint64_t)pair23(st->hb[1], st->hb[0], 0)) &
                       ((LDWC << 4) | (uint64_t)in_ldwc(st->hc[3])));
    uint64_t forceB = S & ((S << 1) | cS) & ~(S >> 1);
    if (next_is_s) forceB &= ~(1ULL << 63);

    uint64_t m = (cc & ~suppA & ~suppC) | forceB | forceC;
    for (long i = 1; i <= 4; i++)
        m = (m & ~(1ULL << i)) | ((uint64_t)gpt2_bnd(st, i) << i);
    m = (m & ~(1ULL << 63)) | ((uint64_t)gpt2_bnd(st, 63) << 63);
    return m;
}

static uint64_t cl100k_mask(const sw_t *st, const tsimd_cls_block_t *b) {
    uint64_t WS = b->sp | b->w | b->c;
    uint64_t SPW = b->sp | b->w;
    uint64_t LDWC = b->l | b->d | b->w | b->c;

    uint8_t p1 = st->hc[0], p2 = st->hc[1];
    uint64_t eq = (b->l & ((b->l << 1) | (uint64_t)(p1 == CLS_L))) |
                  (b->d & ((b->d << 1) | (uint64_t)(p1 == CLS_D))) |
                  (b->sp & ((b->sp << 1) | (uint64_t)(p1 == CLS_SP))) |
                  (b->w & ((b->w << 1) | (uint64_t)(p1 == CLS_W))) |
                  (b->c & ((b->c << 1) | (uint64_t)(p1 == CLS_C))) |
                  (b->o & ((b->o << 1) | (uint64_t)(p1 == CLS_O)));
    uint64_t cc = ~eq;

    uint64_t wscont = WS & ((WS << 1) | (uint64_t)(p1 >= CLS_SP && p1 <= CLS_C));
    if (st->op_carry) wscont &= ~(SPW & (b->c << 1));
    uint64_t supp = wscont;
    supp |= ((b->sp << 1) | (uint64_t)(p1 == CLS_SP)) & (b->l | b->o);
    supp |= ((b->w << 1) | (uint64_t)(p1 == CLS_W)) & b->l;
    supp |= b->c & ((b->o << 1) | (uint64_t)(p1 == CLS_O));

    supp |= b->l & ((b->o << 1) | (uint64_t)(p1 == CLS_O)) &
            ~((b->o << 2) | (uint64_t)(p2 == CLS_O)) &
            ~((b->sp << 2) | (uint64_t)(p2 == CLS_SP));
    uint64_t P = (b->r & (b->e >> 1)) | (b->v & (b->e >> 1)) |
                 (b->l1 & (b->l1 >> 1));
    uint64_t suppC = ((b->ap << 1) | (uint64_t)(st->hb[0] == '\'')) &
                     ((LDWC << 2) | (uint64_t)in_ldwc(st->hc[1])) &
                     (b->suf | P);
    uint64_t forceC = (((b->ap << 2) | (uint64_t)(st->hb[1] == '\'')) &
                       ((b->suf << 1) | (uint64_t)suf1(st->hb[0], 1)) &
                       ((LDWC << 3) | (uint64_t)in_ldwc(st->hc[2]))) |
                      (((b->ap << 3) | (uint64_t)(st->hb[2] == '\'')) &
                       ((P << 2) | (uint64_t)pair23(st->hb[1], st->hb[0], 1)) &
                       ((LDWC << 4) | (uint64_t)in_ldwc(st->hc[3])));

    uint64_t cont3 = b->d & ((b->d << 1) | (uint64_t)(st->d1 != 0)) &
                     ((b->d << 2) | (uint64_t)(st->d2 != 0) |
                      (uint64_t)(st->d1 != 0) << 1);
    uint64_t forceD = b->d & ~((b->d << 1) | (uint64_t)(st->d1 != 0));
    if (st->d1 && (b->d & 1u)) {
        unsigned j0 = (3 - st->d_phase) % 3;
        uint64_t seed_ok =
            (j0 == 0) ? 1u : (j0 == 1) ? 3u : 7u;
        if ((b->d & seed_ok) == seed_ok) forceD |= 1ULL << j0;
    }
    for (int k = 0; k < 22; k++)
        forceD |= (forceD << 3) & cont3;

    {
        uint64_t carry_mask = b->c | (uint64_t)(st->op_carry != 0);
        uint64_t cstarts = b->c & ~(b->c << 1) &
                           ((b->o << 1) | (uint64_t)(p1 == CLS_O));
        uint64_t p = (cstarts | (uint64_t)(st->op_carry != 0)) & carry_mask;

        for (int k = 0; k < 6; k++) {
            p |= p << (1 << k);
            p &= carry_mask;
        }
        uint64_t force_abs = SPW & (p << 1);
        if (st->op_carry) force_abs |= SPW & 1u;
        forceD |= force_abs;
    }

    uint64_t m = (cc & ~supp & ~suppC) | forceC | forceD;
    for (long i = 1; i <= 4; i++)
        m = (m & ~(1ULL << i)) | ((uint64_t)cl100k_bnd(st, i) << i);
    m = (m & ~(1ULL << 63)) | ((uint64_t)cl100k_bnd(st, 63) << 63);
    return m;
}

static uint64_t o200k_mask(const sw_t *st, const tsimd_cls_block_t *b) {
    uint64_t WS = b->sp | b->w | b->c;
    uint64_t SPW = b->sp | b->w;

    uint8_t p1 = st->hc[0];
    uint64_t eq = (b->l & ((b->l << 1) | (uint64_t)(p1 == CLS_L))) |
                  (b->d & ((b->d << 1) | (uint64_t)(p1 == CLS_D))) |
                  (b->sp & ((b->sp << 1) | (uint64_t)(p1 == CLS_SP))) |
                  (b->w & ((b->w << 1) | (uint64_t)(p1 == CLS_W))) |
                  (b->c & ((b->c << 1) | (uint64_t)(p1 == CLS_C))) |
                  (b->o & ((b->o << 1) | (uint64_t)(p1 == CLS_O)));
    uint64_t cc = ~eq;

    uint64_t wscont = WS & ((WS << 1) | (uint64_t)(p1 >= CLS_SP && p1 <= CLS_C));
    if (st->op_carry) wscont &= ~(SPW & (b->c << 1));
    uint64_t supp = wscont;
    supp |= ((b->sp << 1) | (uint64_t)(p1 == CLS_SP)) & (b->l | b->o);
    supp |= ((b->w << 1) | (uint64_t)(p1 == CLS_W)) & b->l;
    supp |= b->c & ((b->o << 1) | (uint64_t)(p1 == CLS_O));
    supp |= b->l & ((b->o << 1) | (uint64_t)(p1 == CLS_O)) &
            ~((b->o << 2) | (uint64_t)(st->hc[1] == CLS_O)) &
            ~((b->sp << 2) | (uint64_t)(st->hc[1] == CLS_SP));

    uint64_t pure_lc = b->lc & ~b->lu;
    uint64_t plc_prev = (pure_lc << 1) |
        (uint64_t)((st->hcase[0] & 2) && !(st->hcase[0] & 1));
    uint64_t caseB = (b->l & ~b->lc) & plc_prev;

    uint64_t cont3 = b->d & ((b->d << 1) | (uint64_t)(st->d1 != 0)) &
                     ((b->d << 2) | (uint64_t)(st->d2 != 0) |
                      (uint64_t)(st->d1 != 0) << 1);
    uint64_t forceD = b->d & ~((b->d << 1) | (uint64_t)(st->d1 != 0));
    if (st->d1 && (b->d & 1u)) {
        unsigned j0 = (3 - st->d_phase) % 3;
        uint64_t seed_ok =
            (j0 == 0) ? 1u : (j0 == 1) ? 3u : 7u;
        if ((b->d & seed_ok) == seed_ok) forceD |= 1ULL << j0;
    }
    for (int k = 0; k < 22; k++)
        forceD |= (forceD << 3) & cont3;

    {
        uint64_t absorb = b->c | b->sl;
        uint64_t carry_mask = absorb | (uint64_t)(st->op_carry != 0);
        uint64_t cstarts = absorb & ~(absorb << 1) &
                           ((b->o << 1) | (uint64_t)(p1 == CLS_O));
        uint64_t p = (cstarts | (uint64_t)(st->op_carry != 0)) & carry_mask;

        for (int k = 0; k < 6; k++) {
            p |= p << (1 << k);
            p &= carry_mask;
        }
        uint64_t force_abs = SPW & (p << 1);
        if (st->op_carry) force_abs |= SPW & 1u;
        forceD |= force_abs;
        supp |= b->sl & (p >> 1);
    }

    uint64_t m = (cc & ~supp) | caseB | forceD;
    for (long i = 1; i <= 4; i++)
        m = (m & ~(1ULL << i)) | ((uint64_t)o200k_bnd(st, i) << i);
    m = (m & ~(1ULL << 63)) | ((uint64_t)o200k_bnd(st, 63) << 63);
    return m;
}

static int push_span(sw_t *st, size_t s, size_t e) {
    int rc;
    if (st->fn) {
        rc = st->fn(st->ud, s, e);
    } else {
        if (e > UINT32_MAX) {
            fprintf(stderr, "tsimd: input exceeds the span-array limit "
                            "(UINT32_MAX); use the streaming API\n");
            rc = -1;
        } else {
            rc = tsimd_span_array_push(st->out, (uint32_t)s, (uint32_t)e);
        }
    }
    if (rc != 0) st->err = 1;
    return rc;
}

static void seek(sw_t *st, size_t pos) {
    st->base = pos;
    st->span_start = pos;
    st->cache_valid = 0;
    for (int k = 0; k < 4; k++) {
        long idx = (long)pos - 1 - k;
        if (idx >= 0) {
            st->hb[k] = st->data[(size_t)idx];

            size_t start = (size_t)idx;
            while (start > 0 && (st->data[start] & 0xC0) == 0x80) start--;
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t adv = tsimd_utf8_next(st->data + start, st->len - start,
                                         &cp, &cls);
            if (adv == 0 || start + adv <= (size_t)idx) {
                adv = tsimd_utf8_next(st->data + (size_t)idx,
                                      st->len - (size_t)idx, &cp, &cls);
                if (adv == 0) {
                    cls = TSIMD_CLASS_OTHER;
                    cp = st->data[(size_t)idx];
                }
            }
            st->hcase[k] = tsimd_class_case(cp);
            switch (cls) {
                case TSIMD_CLASS_LETTER: st->hc[k] = CLS_L; break;
                case TSIMD_CLASS_NUMBER: st->hc[k] = CLS_D; break;
                case TSIMD_CLASS_SPACE:

                    if (st->hb[k] == 0x20) st->hc[k] = CLS_SP;
                    else if (st->hb[k] == '\r' || st->hb[k] == '\n')
                        st->hc[k] = CLS_C;
                    else st->hc[k] = CLS_W;
                    break;
                default:                st->hc[k] = CLS_O; break;
            }
        } else {
            st->hb[k] = 0;
            st->hc[k] = CLS_L;
            st->hcase[k] = 0;
        }
    }

    {
        size_t off = pos;
        size_t t = 0;
        while (off > 0) {
            size_t start = off - 1;
            while (start > 0 && (st->data[start] & 0xC0) == 0x80) start--;

            size_t ncont = off - 1 - start;
            uint8_t lb = st->data[start];
            size_t elen = (lb & 0xC0) == 0xC0 ? 2
                          : (lb & 0xE0) == 0xE0 ? 3
                          : (lb & 0xF8) == 0xF0 ? 4 : 1;
            if (lb < 0x80 || ncont + 1 != elen) break;
            uint32_t cp;
            tsimd_char_class_t cls;
            tsimd_utf8_next(st->data + start, st->len - start, &cp, &cls);
            if (cls != TSIMD_CLASS_NUMBER) break;
            t++;
            off = start;
        }
        st->d_open = (t > 0);
        st->d_phase = (unsigned)(t % 3);
        st->d1 = (t >= 1);
        st->d2 = (t >= 2);
    }

    {
        size_t k = pos;
        st->op_carry = 0;
        if (k > 0 && (st->data[k - 1] & 0x80) == 0 &&
            tsimd_class_byte(st->data[k - 1]) == CLS_C) {
            while (k > 0 && (st->data[k - 1] & 0x80) == 0 &&
                   tsimd_class_byte(st->data[k - 1]) == CLS_C)
                k--;
            st->op_carry = (k > 0 && (st->data[k - 1] & 0x80) == 0 &&
                            tsimd_class_byte(st->data[k - 1]) == CLS_O);
        }
    }
}

static void advance(sw_t *st, const tsimd_cls_block_t *blk) {
    uint8_t p1_prev = st->hc[0];
    for (int k = 0; k < 4; k++) {
        size_t idx = st->base + 63 - (size_t)k;
        st->hb[k] = st->data[idx];
        st->hc[k] = tsimd_class_byte(st->data[idx]);
        uint8_t byte = st->data[idx];
        st->hcase[k] = (byte >= 'A' && byte <= 'Z') ? 1
                     : (byte >= 'a' && byte <= 'z') ? 2 : 0;
    }
    uint64_t nd = ~blk->d;

    uint64_t r = nd ? (uint64_t)__builtin_clzll(nd) : 64;
    st->d_phase = (unsigned)((r == 64) ? (st->d_phase + 64u) % 3u : r % 3u);
    st->d_open = (int)((blk->d >> 63) & 1u);
    st->d1 = st->d_open;
    st->d2 = (int)((blk->d >> 62) & 1u);

    {
        uint64_t carry_mask = blk->c | (uint64_t)(st->op_carry != 0);
        uint64_t cstarts = blk->c & ~(blk->c << 1) &
                           ((blk->o << 1) |
                            (uint64_t)(p1_prev == CLS_O));
        uint64_t p = (cstarts | (uint64_t)(st->op_carry != 0)) & carry_mask;

        for (int k = 0; k < 6; k++) {
            p |= p << (1 << k);
            p &= carry_mask;
        }
        st->op_carry = (int)((p >> 63) & 1u);
    }
    st->base += TSIMD_BLOCK;
}

static int scalar_fallback(sw_t *st) {
    size_t i = st->span_start;
    while (i < st->len) {
        if (st->err) return 1;
        size_t e = st->is_cl100k ? tsimd_match_cl100k(st->data, st->len, i)
                   : st->is_o200k ? tsimd_match_o200k(st->data, st->len, i)
                                  : tsimd_match_gpt2(st->data, st->len, i);
        (void)push_span(st, i, e);
        i = e;
        if (e + TSIMD_BLOCK <= st->len) {
            tsimd_cls_block_t blk;
            st->classify(st->data + e, st->ci, &blk);
            if (!blk.hi &&
                !(st->is_o200k &&
                  (blk.ap & ((blk.l >> 1) | blk.l | (blk.l << 1))))) {
                seek(st, e);
                return 0;
            }
        }
    }
    return 1;
}

static const tsimd_cls_block_t *ensure_block(sw_t *st, size_t base) {
    if (st->cache_valid && st->cache_base == base) return &st->cache;
    if (st->len - base < TSIMD_BLOCK) return NULL;
    st->classify(st->data + base, st->ci, &st->cache);
    st->cache_valid = 1;
    st->cache_base = base;
    return &st->cache;
}

static int ws_scan(sw_t *st, size_t q, size_t *e_out, size_t *lastc_out,
                   int *anyc_out) {
    size_t lastc = q;
    int anyc = 0;
    size_t pos = q & ~(size_t)(TSIMD_BLOCK - 1);

    while (st->len - pos >= TSIMD_BLOCK) {
        const tsimd_cls_block_t *cur = ensure_block(st, pos);
        if (cur->hi) return -1;
        size_t rel = (pos > q) ? 0 : q - pos;
        uint64_t ws = cur->sp | cur->w | cur->c;
        uint64_t geq = (rel == 0) ? ~0ULL : ~((1ULL << rel) - 1ULL);
        uint64_t nz = ~ws & geq;
        if (nz) {
            size_t k = (size_t)__builtin_ctzll(nz);
            uint64_t below = (k >= 64) ? ~0ULL : ((1ULL << k) - 1ULL);
            uint64_t c_here = cur->c & geq & below;
            if (c_here) {
                anyc = 1;
                lastc = pos + 64 - (size_t)__builtin_clzll(c_here);
            }
            *e_out = pos + k;
            *lastc_out = lastc;
            *anyc_out = anyc;
            return 0;
        }
        uint64_t c_adv = cur->c & geq;
        if (c_adv) {
            anyc = 1;
            lastc = pos + 64 - (size_t)__builtin_clzll(c_adv);
        }
        pos += TSIMD_BLOCK;
    }

    size_t start = (pos > q) ? pos : q;
    for (size_t k = start; k < st->len; k++) {
        uint8_t byte = st->data[k];
        if (byte & 0x80) return -1;
        uint8_t c = tsimd_class_byte(byte);
        if (c < CLS_SP || c > CLS_C) {
            *e_out = k;
            *lastc_out = lastc;
            *anyc_out = anyc;
            return 0;
        }
        if (c == CLS_C) { anyc = 1; lastc = k + 1; }
    }
    *e_out = st->len;
    *lastc_out = lastc;
    *anyc_out = anyc;
    return 0;
}

static size_t ws_patch(sw_t *st, size_t q) {
    for (;;) {
        size_t e, lastc;
        int anyc;
        if (ws_scan(st, q, &e, &lastc, &anyc) != 0)
            return (size_t)-1;
        if (anyc) {
            push_span(st, q, lastc);
            q = lastc;
            if (q >= e || e >= st->len) return q;
            continue;
        }
        if (e >= st->len) return q;
        if (e - q >= 2) {
            push_span(st, q, e - 1);
            q = e - 1;
            continue;
        }
        return q;
    }
}

static size_t patch_chain(sw_t *st, const tsimd_cls_block_t *blk, size_t base,
                          size_t p) {
    size_t cur = p;
    for (;;) {
        uint8_t c = tsimd_class_byte(st->data[cur]);
        if (c < CLS_SP || c > CLS_C) return cur;

        if (cur < base + 64) {
            size_t rel = cur - base;
            uint64_t geq = (rel == 0) ? ~0ULL : ~((1ULL << rel) - 1ULL);
            uint64_t ws = blk->sp | blk->w | blk->c;
            uint64_t nz = ~ws & geq;
            if (nz) {
                size_t k = (size_t)__builtin_ctzll(nz);
                uint64_t below = (k >= 64) ? ~0ULL : ((1ULL << k) - 1ULL);
                if (!(blk->c & geq & below)) {
                    if (k - rel == 1)
                        return cur;
                }
            }
        }
        size_t np = ws_patch(st, cur);
        if (np == (size_t)-1) return (size_t)-1;
        if (np == cur) return cur;
        cur = np;
        (void)base;
        if (cur >= base + 64) return cur;
    }
}

int tsimd_pretokenize_simd_ex(tsimd_pretok_scheme_t scheme,
                             const uint8_t *data, size_t len,
                             tsimd_span_fn fn, void *ud,
                             tsimd_span_array_t *out) {
    sw_t st;
    memset(&st, 0, sizeof(st));
    st.data = data;
    st.len = len;
    st.is_cl100k = (scheme == TSIMD_PRETOK_CL100K);
    st.is_o200k = (scheme == TSIMD_PRETOK_O200K);
    st.ci = st.is_cl100k || st.is_o200k;
    st.classify = tsimd_simd_classify();
    st.out = out;
    st.fn = fn;
    st.ud = ud;
    seek(&st, 0);

    while (st.span_start < len) {
        if (st.err) return -1;
        if (len - st.base < TSIMD_BLOCK) {
            int sf = scalar_fallback(&st);
            if (st.err) return -1;
            if (sf) return 0;
            continue;
        }
        const tsimd_cls_block_t *blk = ensure_block(&st, st.base);
        if (!blk || blk->hi) {
            st.cache_valid = 0;
            int sf = scalar_fallback(&st);
            if (st.err) return -1;
            if (sf) return 0;
            continue;
        }
        tsimd_cls_block_t blk_local = *blk;

        if (st.is_o200k &&
            (blk_local.ap & ((blk_local.l >> 1) | blk_local.l |
                             (blk_local.l << 1)))) {
            st.cache_valid = 0;
            int sf = scalar_fallback(&st);
            if (st.err) return -1;
            if (sf) return 0;
            continue;
        }

        if ((st.is_cl100k || st.is_o200k) && st.span_start == st.base) {
            size_t np0 = patch_chain(&st, &blk_local, st.base, st.base);
            if (np0 == (size_t)-1) {
                int sf = scalar_fallback(&st);
                if (st.err) return -1;
                if (sf) return 0;
                continue;
            }
            if (np0 != st.base) {
                st.span_start = np0;
                if (np0 - st.base >= TSIMD_BLOCK) {
                    seek(&st, np0);
                    continue;
                }

            }
        }

        int next_is_s = 1;
        if (!st.is_cl100k && st.base + TSIMD_BLOCK < len) {
            uint8_t nc = data[st.base + TSIMD_BLOCK];
            if (nc < 0x80) {
                uint8_t ncls = tsimd_class_byte(nc);
                next_is_s = (ncls >= CLS_SP && ncls <= CLS_C);
            } else {
                uint32_t ncp;
                tsimd_char_class_t nxc;
                size_t nadv = tsimd_utf8_next(data + st.base + TSIMD_BLOCK,
                                              len - st.base - TSIMD_BLOCK,
                                              &ncp, &nxc);
                next_is_s = (nadv > 0 && nxc == TSIMD_CLASS_SPACE);
            }
        }

        uint64_t m = st.is_cl100k ? cl100k_mask(&st, &blk_local)
                     : st.is_o200k ? o200k_mask(&st, &blk_local)
                                   : gpt2_mask(&st, &blk_local, next_is_s);
        {
            static const char *dbg = NULL;
            static int dbg_on = -1;
            if (dbg_on == -1) {
                dbg = getenv("TSIMD_SPAN_DBG");
                dbg_on = dbg ? 1 : 0;
            }
            if (dbg_on && st.base >= 60 && st.base < 100) {
                fprintf(stderr, "BLK base=%zu m=%016llx d=%016llx "
                        "span_start=%zu d_phase=%u d1=%d d2=%d\n",
                        st.base, (unsigned long long)m,
                        (unsigned long long)blk_local.d, st.span_start,
                        st.d_phase, st.d1, st.d2);
            }
        }
        if (st.span_start > st.base)
            m &= (st.span_start - st.base >= TSIMD_BLOCK)
                     ? 0ULL
                     : ~(((1ULL << (st.span_start - st.base)) - 1ULL));

        int seeked = 0;
        while (m) {
            int bit = __builtin_ctzll(m);
            m &= m - 1ULL;
            size_t p = st.base + (size_t)bit;
            if (p == st.span_start) continue;
            push_span(&st, st.span_start, p);
            st.span_start = p;
            if (st.is_cl100k || st.is_o200k) {
                size_t np = patch_chain(&st, &blk_local, st.base, p);
                if (np == (size_t)-1) {
                    int sf = scalar_fallback(&st);
                    if (st.err) return -1;
                    if (sf) return 0;
                    seeked = 1;
                    break;
                }
                if (np != p) {
                    st.span_start = np;
                    if (np - st.base >= TSIMD_BLOCK) {
                        seek(&st, np);
                        seeked = 1;
                        break;
                    }

                    if (np - st.base >= TSIMD_BLOCK - 1)
                        m = 0;
                    else
                        m &= ~(((2ULL << (np - st.base)) - 1ULL));
                }
            }
        }
        if (seeked) continue;
        advance(&st, &blk_local);
    }
    return st.err ? -1 : 0;
}

int tsimd_pretokenize_simd(tsimd_pretok_scheme_t scheme,
                          const uint8_t *data, size_t len,
                          tsimd_span_array_t *out) {
    return tsimd_pretokenize_simd_ex(scheme, data, len, NULL, NULL, out);
}
