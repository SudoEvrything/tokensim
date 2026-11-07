#include <stdio.h>
#include <string.h>

#include "simd.h"

int main(void) {
    int supported = tsimd_simd_level_supported(TSIMD_SIMD_AVX512);
    int probe = tsimd_simd_avx512_probe();
    int level = (int)tsimd_simd_level();

    printf("  avx512: libgcc-supported=%d probe=%d level=%d\n", supported,
           probe, level);

    if (probe != 0 && probe != 1) {
        fprintf(stderr, "FAIL: probe returned %d\n", probe);
        return 1;
    }
    if (!supported && probe != 0) {
        fprintf(stderr,
                "FAIL: the probe executed avx512 where the CPU/libgcc "
                "reports no support\n");
        return 1;
    }
    if (probe == 1 && !supported) {
        fprintf(stderr, "FAIL: probe=1 without libgcc support\n");
        return 1;
    }
    if (level == (int)TSIMD_SIMD_AVX512 && probe != 1) {
        fprintf(stderr,
                "FAIL: the dispatch selected avx512 but the probe did "
                "not pass\n");
        return 1;
    }
    if (probe == 0 && supported) {
        printf("  avx512: the CPU reports support but the probe faults "
               "- the tolerance downgrade is the correct state\n");
    }
    printf("test_avx512_probe: OK\n");
    return 0;
}
