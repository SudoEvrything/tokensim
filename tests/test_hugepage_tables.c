#include "allocator.h"
#include "bpe.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HUGEPG ((size_t)2 << 20)

static int huge_aligned(const void *p) {
    return ((uintptr_t)p & (HUGEPG - 1)) == 0;
}

static int test_helper_contract(void) {
    void *small = tsimd_alloc_huge(1024);
    if (!small) return 1;
    memset(small, 0xAB, 1024);

    void *big = tsimd_alloc_huge((size_t)3 << 20);
    if (!big || !huge_aligned(big)) return 2;
    memset(big, 0xCD, (size_t)3 << 20);

    void *bz = tsimd_alloc_huge_zero((size_t)3 << 20);
    if (!bz || !huge_aligned(bz)) return 3;
    const unsigned char *z = bz;
    for (size_t i = 0; i < ((size_t)3 << 20); i += 4096)
        if (z[i] != 0) return 4;

    unsigned char *w = big;
    w[0] = 1;
    w[((size_t)3 << 20) - 1] = 2;
    if (w[0] != 1 || w[((size_t)3 << 20) - 1] != 2) return 5;

    tsimd_xfree(small);
    tsimd_xfree(big);
    tsimd_xfree(bz);
    printf("  helper: small passthrough, 2 MiB alignment, zero fill\n");
    return 0;
}

static void unit6(uint8_t *b, uint32_t i) {
    b[0] = (uint8_t)i;
    b[1] = (uint8_t)(i >> 8);
    b[2] = 0x5A;
    b[3] = (uint8_t)(i * 7);
    b[4] = 0xC3;
    b[5] = 0x11;
}

static int test_plong_tables(void) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 1024) != 0) return 1;
    c.lmin = 5;
    uint8_t b[6];
    uint32_t ids[4] = {1, 2, 3, 4};
    unit6(b, 0);
    tsimd_plong_insert(&c, b, 6, ids, 4);

    size_t ltab_bytes =
        ((size_t)c.lmask + 1) * sizeof(tsimd_plong_entry_t);
    if (!c.ltab || !c.larena || !c.lids) return 2;
    if (ltab_bytes >= HUGEPG && !huge_aligned(c.ltab)) return 3;
    if (c.lcap >= HUGEPG && !huge_aligned(c.larena)) return 4;
    if (c.lids_cap * sizeof(uint32_t) >= HUGEPG &&
        !huge_aligned(c.lids))
        return 5;

    const uint32_t N = 140000;
    for (uint32_t i = 0; i < N; i++) {
        unit6(b, i);
        uint32_t q[4] = {i & 0xFFFF, 0x77, 0x12, 0x34};
        tsimd_plong_insert(&c, b, 6, q, 4);
    }
    if (c.lbits <= 17) return 6;
    if (!huge_aligned(c.ltab)) return 7;
    uint32_t n = 0;
    unit6(b, N - 1);
    const uint32_t *hit = tsimd_plong_probe_ex(&c, b, 6, &n);
    if (!hit || n != 4 || hit[0] != ((N - 1) & 0xFFFF)) return 8;

    tsimd_pretok_cache_free(&c);
    printf("  plong: ltab/larena/lids 2 MiB aligned at ensure and grow\n");
    return 0;
}

int main(void) {
    int rc = test_helper_contract();
    if (rc) {
        fprintf(stderr, "FAIL hugepage helper contract (%d)\n", rc);
        return 1;
    }
    rc = test_plong_tables();
    if (rc) {
        fprintf(stderr, "FAIL plong hugepage tables (%d)\n", rc);
        return 1;
    }
    return 0;
}
