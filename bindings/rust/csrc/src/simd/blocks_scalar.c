#include "simd_internal.h"

void tsimd_classify_block_scalar(const uint8_t *data, int case_insensitive,
                                tsimd_cls_block_t *blk) {
    uint64_t l = 0, d = 0, sp = 0, w = 0, c = 0, o = 0, hi = 0;
    uint64_t ap = 0, suf = 0, r = 0, v = 0, l1 = 0, e = 0;
    uint64_t lu = 0, lc = 0, sl = 0;

    for (int i = 0; i < TSIMD_CLS_BLOCK_BYTES; i++) {
        uint8_t b = data[i];
        uint64_t bit = 1ULL << i;
        if (b & 0x80) {
            hi |= bit;
            continue;
        }
        uint8_t x = case_insensitive ? (uint8_t)(b | 0x20) : b;
        switch (tsimd_class_byte(b)) {
            case CLS_L:
                l |= bit;
                if (b >= 'A' && b <= 'Z') lu |= bit;
                else lc |= bit;
                break;
            case CLS_D: d |= bit; break;
            case CLS_SP: sp |= bit; break;
            case CLS_W: w |= bit; break;
            case CLS_C: c |= bit; break;
            default: o |= bit; break;
        }
        if (b == '\'') ap |= bit;
        if (b == '/') sl |= bit;
        if (x == 's' || x == 't' || x == 'm' || x == 'd') suf |= bit;
        if (x == 'r') r |= bit;
        if (x == 'v') v |= bit;
        if (x == 'l') l1 |= bit;
        if (x == 'e') e |= bit;
    }
    blk->l = l; blk->d = d; blk->sp = sp; blk->w = w; blk->c = c;
    blk->o = o; blk->hi = hi;
    blk->ap = ap; blk->suf = suf; blk->r = r; blk->v = v; blk->l1 = l1;
    blk->e = e;
    blk->lu = lu; blk->lc = lc; blk->sl = sl;
}
