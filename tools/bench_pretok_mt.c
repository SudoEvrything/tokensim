







#include "pretokenizer.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

typedef struct {
    const uint8_t *data;
    size_t len;
    tsimd_pretok_scheme_t scheme;
    size_t spans;
    double secs;
} slice_t;

static atomic_int next_slice;
static size_t n_slices_total;

static void *worker(void *arg) {
    slice_t *slices = arg;
    for (;;) {
        int i = atomic_fetch_add(&next_slice, 1);
        if (i < 0 || (size_t)i > n_slices_total ||
            slices[(size_t)i].len == 0)
            break;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        tsimd_span_array_t spans;
        tsimd_span_array_init(&spans);
        if (tsimd_pretokenize(slices[i].scheme, slices[i].data,
                              slices[i].len, &spans) != 0) {
            fprintf(stderr, "pretokenize failed at slice %d\n", i);
            exit(1);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        slices[i].spans = spans.count;
        slices[i].secs = (double)(t1.tv_sec - t0.tv_sec) +
                         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        tsimd_span_array_free(&spans);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s corpus [scheme=gpt2] [threads...]\n",
                argv[0]);
        return 1;
    }
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (argc > 2 && strcmp(argv[2], "o200k") == 0) scheme = TSIMD_PRETOK_O200K;
    else if (argc > 2 && strcmp(argv[2], "cl100k") == 0) scheme = TSIMD_PRETOK_CL100K;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) return 1;
    off_t flen = lseek(fd, 0, SEEK_END);
    size_t total = (size_t)flen;
    const uint8_t *data = mmap(NULL, total, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) return 1;
    madvise((void *)data, total, MADV_WILLNEED);

    int max_t = (argc > 3) ? atoi(argv[3]) : 16;
    printf("%s: %zu MiB, scheme=%s\n", argv[1], total >> 20,
           scheme == TSIMD_PRETOK_O200K ? "o200k"
           : scheme == TSIMD_PRETOK_CL100K ? "cl100k" : "gpt2");

    size_t slice_len = 1u << 20;
    size_t n_slices = total / slice_len;
    n_slices_total = n_slices;
    slice_t *slices = calloc(n_slices + 1, sizeof(slice_t));
    for (size_t i = 0; i < n_slices; i++) {
        slices[i].data = data + i * slice_len;
        slices[i].len = slice_len;
        slices[i].scheme = scheme;
    }

    for (int t = 1; t <= max_t; t = (t < 4) ? t + 1 : (t < 10) ? t + 2 : t + 4) {
        if (t > max_t) t = max_t;
        for (size_t i = 0; i < n_slices; i++) {
            slices[i].spans = 0;
            slices[i].secs = 0;
        }
        atomic_store(&next_slice, 0);
        pthread_t th[64];
        for (int i = 0; i < t - 1; i++)
            pthread_create(&th[i], NULL, worker, slices);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        worker(slices);
        for (int i = 0; i < t - 1; i++) pthread_join(th[i], NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double wall = (double)(t1.tv_sec - t0.tv_sec) +
                      (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        size_t spans = 0;
        for (size_t i = 0; i < n_slices; i++) spans += slices[i].spans;
        double mibs = (double)total / (1024.0 * 1024.0) / wall;
        printf("threads=%2d: %6.1f MiB/s  (%5.2f/thread)  %zu spans\n",
               t, mibs, mibs / t, spans);
        if (t >= max_t) break;
    }
    munmap((void *)data, total);
    return 0;
}
