#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

extern int tsimd_cpu_core_kind(int cpu);

static int hybrid;
static int hw_threads(void) {
    int n = 0;
    FILE *f = fopen("/sys/devices/system/cpu/online", "rb");
    if (!f) return 0;
    int a = -1, b = -1, c;
    while ((c = getc(f)) != EOF) {
        if (c >= '0' && c <= '9') {
            if (a < 0) a = c - '0';
            else b = b < 0 ? c - '0' : b * 10 + c - '0';
        } else if (c == '-') {
            b = -1;
        } else if (c == ',' || c == '\n') {
            n += b >= 0 ? b - a + 1 : (a >= 0 ? 1 : 0);
            a = b = -1;
        }
    }
    fclose(f);
    return n;
}

int main(void) {
    for (int cpu = 0; cpu < 64; cpu++) {
        int k = tsimd_cpu_core_kind(cpu);
        if (k < 0 || k > 3) {
            fprintf(stderr, "FAIL: cpu %d kind %d out of range\n", cpu, k);
            return 1;
        }
        if (k != 0 && k != 1 && k != 2 && k != 3) return 1;
    }
    int n = hw_threads();
    int kind0 = tsimd_cpu_core_kind(0);
    int kind12 = n > 12 ? tsimd_cpu_core_kind(12) : 0;
    hybrid = kind0 == 2 && kind12 == 1;
    if (!hybrid) {
        printf("test_worker_caps: SKIP hybrid assertions (non-hybrid or "
               "short topology: n=%d kind0=%d kind12=%d)\n",
               n, kind0, kind12);
        return 0;
    }
    for (int cpu = 0; cpu < 12; cpu++) {
        int k = tsimd_cpu_core_kind(cpu);
        if (k != 2 && k != 3) {
            fprintf(stderr, "FAIL: hybrid P cpu %d kind %d\n", cpu, k);
            return 1;
        }
    }
    for (int cpu = 12; cpu < n && cpu < 64; cpu++) {
        if (tsimd_cpu_core_kind(cpu) != 1) {
            fprintf(stderr, "FAIL: hybrid E cpu %d kind %d\n", cpu,
                    tsimd_cpu_core_kind(cpu));
            return 1;
        }
    }
    for (int cpu = 0; cpu < 12; cpu += 2) {
        int a = tsimd_cpu_core_kind(cpu);
        int b = tsimd_cpu_core_kind(cpu + 1);
        if (a != 2 || b != 3) {
            fprintf(stderr, "FAIL: sibling pair (%d,%d) = (%d,%d)\n", cpu,
                    cpu + 1, a, b);
            return 1;
        }
    }
    printf("test_worker_caps: OK (hybrid map verified, %d threads)\n", n);
    return 0;
}
