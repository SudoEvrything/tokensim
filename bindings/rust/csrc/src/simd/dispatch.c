#include "simd.h"
#include "simd_internal.h"

#include <signal.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#define TSIMD_X86 1
#endif

#if defined(TSIMD_X86)
#include <cpuid.h>

static uint64_t tsimd_xgetbv0(void) {
    uint32_t a = 0, d = 0;
    __asm__ __volatile__("xgetbv"
                         : "=a"(a), "=d"(d)
                         : "c"(0));
    return (uint64_t)a | ((uint64_t)d << 32);
}

static int tsimd_cpuid_bit(uint32_t leaf, uint32_t subleaf,
                           uint32_t reg, uint32_t bit) {
    uint32_t a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(leaf, subleaf, &a, &b, &c, &d)) return 0;
    switch (reg) {
        case 0: return (a >> bit) & 1u;
        case 1: return (b >> bit) & 1u;
        case 2: return (c >> bit) & 1u;
        default: return (d >> bit) & 1u;
    }
}

static int tsimd_cpu_avx2_os(void) {

    if (!tsimd_cpuid_bit(1, 0, 2, 27)) return 0;
    if ((tsimd_xgetbv0() & 0x6) != 0x6) return 0;
    return tsimd_cpuid_bit(7, 0, 1, 5);
}

static int tsimd_cpu_avx512_os(void) {

    if ((tsimd_xgetbv0() & 0xE0) != 0xE0) return 0;
    if (!tsimd_cpuid_bit(7, 0, 1, 16)) return 0;
    if (!tsimd_cpuid_bit(7, 0, 1, 30)) return 0;
    if (!tsimd_cpuid_bit(7, 0, 1, 31)) return 0;
    return 1;
}
#endif

static atomic_int g_forced = -1;
static atomic_int g_detected = -1;
static atomic_int g_probe = -1;

static sigjmp_buf probe_jmp;

static void probe_sigill_handler(int sig) {
    (void)sig;
    siglongjmp(probe_jmp, 1);
}

static int detect_level(void) {
    int d = atomic_load_explicit(&g_detected, memory_order_relaxed);
    if (d >= 0) return (tsimd_simd_level_t)d;
#if defined(TSIMD_X86) && defined(__x86_64__)
    if (tsimd_cpu_avx2_os())
        atomic_store_explicit(&g_detected, TSIMD_SIMD_AVX2,
                              memory_order_relaxed);
    else
        atomic_store_explicit(&g_detected, TSIMD_SIMD_SSE2,
                              memory_order_relaxed);
    if (atomic_load_explicit(&g_detected, memory_order_relaxed) ==
        TSIMD_SIMD_AVX2) {
        const char *deny = getenv("TSIMD_SIMD_NO_AVX512");
        const char *legacy = getenv("TSIMD_SIMD_ALLOW_AVX512");
        int denied = (deny && deny[0] == '1') ||
                     (legacy && legacy[0] == '0');
        if (!denied && tsimd_cpu_avx512_os() &&
            tsimd_simd_avx512_probe() == 1)
            atomic_store_explicit(&g_detected, TSIMD_SIMD_AVX512,
                                  memory_order_relaxed);
    }
#else
    atomic_store_explicit(&g_detected, TSIMD_SIMD_SCALAR,
                          memory_order_relaxed);
#endif
    return (tsimd_simd_level_t)
        atomic_load_explicit(&g_detected, memory_order_relaxed);
}

void tsimd_simd_set_forced(int level) {
    atomic_store_explicit(&g_forced, level, memory_order_relaxed);
}

int tsimd_simd_level_supported(tsimd_simd_level_t level) {
    if (level == TSIMD_SIMD_SCALAR) return 1;
#if !defined(TSIMD_X86)
    return 0;
#else
    if (level == TSIMD_SIMD_SSE2) return 1;
    if (level == TSIMD_SIMD_AVX2) return tsimd_cpu_avx2_os();
    if (level == TSIMD_SIMD_AVX512)
        return tsimd_cpu_avx512_os() &&
               tsimd_simd_avx512_probe() == 1;
    return 0;
#endif
}

int tsimd_simd_avx512_probe(void) {
    static int cached = -1;
    int c = atomic_load_explicit(&g_probe, memory_order_relaxed);
    if (c >= 0) return c;
#if defined(TSIMD_X86) && defined(__x86_64__)
    if (!tsimd_cpu_avx512_os()) {
        atomic_store_explicit(&g_probe, 0, memory_order_relaxed);
        return 0;
    }
    void (*old)(int) = signal(SIGILL, probe_sigill_handler);
    if (sigsetjmp(probe_jmp, 1) == 0) {
        int ok = tsimd_avx512_selftest();
        signal(SIGILL, old);
        atomic_store_explicit(&g_probe, ok, memory_order_relaxed);
        return ok;
    }
    signal(SIGILL, old);
    atomic_store_explicit(&g_probe, 0, memory_order_relaxed);
    return 0;
#else
    atomic_store_explicit(&g_probe, 0, memory_order_relaxed);
    return 0;
#endif
}

tsimd_simd_level_t tsimd_simd_level(void) {
    static int env_done = 0;
    if (!env_done) {
        env_done = 1;
        const char *env = getenv("TSIMD_SIMD_FORCE");
        if (env) {
            if (strcmp(env, "scalar") == 0)
                atomic_store_explicit(&g_forced, TSIMD_SIMD_SCALAR,
                                      memory_order_relaxed);
            else if (strcmp(env, "sse2") == 0)
                atomic_store_explicit(&g_forced, TSIMD_SIMD_SSE2,
                                      memory_order_relaxed);
            else if (strcmp(env, "avx2") == 0)
                atomic_store_explicit(&g_forced, TSIMD_SIMD_AVX2,
                                      memory_order_relaxed);
            else if (strcmp(env, "avx512") == 0)
                atomic_store_explicit(&g_forced, TSIMD_SIMD_AVX512,
                                      memory_order_relaxed);
        }
    }
    int level;
    int forced = atomic_load_explicit(&g_forced, memory_order_relaxed);
    if (forced >= 0) {

        level = tsimd_simd_level_supported((tsimd_simd_level_t)forced)
                    ? forced
                    : (int)detect_level();
    } else {
        level = (int)detect_level();
    }
    return (tsimd_simd_level_t)level;
}

tsimd_classify_fn_t tsimd_simd_classify(void) {
    switch (tsimd_simd_level()) {
        case TSIMD_SIMD_AVX512: return tsimd_classify_block_avx512;
        case TSIMD_SIMD_AVX2: return tsimd_classify_block_avx2;
        case TSIMD_SIMD_SSE2: return tsimd_classify_block_sse2;
        default: return tsimd_classify_block_scalar;
    }
}
