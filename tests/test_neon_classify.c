#include "simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void tsimd_classify_block_scalar(const uint8_t *data,
                                        int case_insensitive,
                                        tsimd_cls_block_t *blk);
extern void tsimd_classify_block_neon(const uint8_t *data,
                                      int case_insensitive,
                                      tsimd_cls_block_t *blk);

static int g_fail;
static int checks;

static void cmp_block(const tsimd_cls_block_t *a, const tsimd_cls_block_t *b,
                      int ci, unsigned seed) {
    static const char *names[] = {"l",  "d",   "sp", "w",   "c",
                                  "o",  "hi",  "ap", "suf", "r",
                                  "v",  "l1",  "e",  "lu",  "lc",
                                  "sl"};
    const uint64_t *pa = (const uint64_t *)a;
    const uint64_t *pb = (const uint64_t *)b;
    for (size_t f = 0; f < sizeof(names) / sizeof(names[0]); f++) {
        checks++;
        if (pa[f] != pb[f]) {
            fprintf(stderr,
                    "FAIL field %s ci=%d seed=%u scalar=%016llx neon=%016llx\n",
                    names[f], ci, seed, (unsigned long long)pa[f],
                    (unsigned long long)pb[f]);
            g_fail = 1;
        }
    }
}

static uint32_t xorshift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

int main(void) {
    static const struct {
        const char *name;
        uint8_t bytes[64];
    } fixed[] = {
        {"all 0-63", {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                      42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                      56, 57, 58, 59, 60, 61, 62, 63}},
        {"64-127", {64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                    76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
                    88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                    124, 125, 126, 127}},
        {"128-191",
         {128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
          142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
          156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
          170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
          184, 185, 186, 187, 188, 189, 190, 191}},
        {"192-255",
         {192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
          206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
          220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
          234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
          248, 249, 250, 251, 252, 253, 254, 255}},
        {"whitespace and letters",
         {' ', '\t', '\n', 11, 12, '\r', 'a', 'z', 'A', 'Z', '0', '9', '\'',
          '/', 's', 't', 'm', 'd', 'r', 'v', 'l', 'e', 'S', 'T', 'M', 'D',
          'R', 'V', 'L', 'E', 0x80, 0xff, ' ', '\t', '\n', 11, 12, '\r', 'a',
          'z', 'A', 'Z', '0', '9', '\'', '/', 's', 't', 'm', 'd', 'r', 'v',
          'l', 'e', 'S', 'T', 'M', 'D', 'R', 'V', 'L', 'E', 0x80, 0xff, ' '}},
    };

    for (size_t t = 0; t < sizeof(fixed) / sizeof(fixed[0]); t++) {
        for (int ci = 0; ci <= 1; ci++) {
            tsimd_cls_block_t a, b;
            tsimd_classify_block_scalar(fixed[t].bytes, ci, &a);
            tsimd_classify_block_neon(fixed[t].bytes, ci, &b);
            checks++;
            if (memcmp(&a, &b, sizeof(a)) != 0) {
                fprintf(stderr, "FAIL fixed %s ci=%d\n", fixed[t].name, ci);
                cmp_block(&a, &b, ci, (unsigned)t);
            }
        }
    }

    uint32_t seed = 0x12345678u;
    for (int round = 0; round < 512; round++) {
        uint8_t buf[64];
        for (int i = 0; i < 64; i++) {
            uint32_t r = xorshift(&seed);
            switch (round % 4) {
                case 0: buf[i] = (uint8_t)r; break;
                case 1: buf[i] = (uint8_t)(r & 0x7f); break;
                case 2:
                    buf[i] = (uint8_t)" \t\n\r'aZ09/smdrvle"[(r >> 8) % 19];
                    break;
                default: buf[i] = (uint8_t)(0x80 | (r & 0x3f)); break;
            }
        }
        for (int ci = 0; ci <= 1; ci++) {
            tsimd_cls_block_t a, b;
            tsimd_classify_block_scalar(buf, ci, &a);
            tsimd_classify_block_neon(buf, ci, &b);
            checks++;
            if (memcmp(&a, &b, sizeof(a)) != 0)
                cmp_block(&a, &b, ci, seed);
        }
    }

    printf("test_neon_classify: %d checks, %s\n", checks,
           g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
