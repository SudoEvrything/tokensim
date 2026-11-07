#include "allocator.h"

#include <sys/mman.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tsimd_arena_chunk {
    tsimd_arena_chunk_t *next;
    size_t used;
    size_t cap;

};

#define TSIMD_ARENA_FIRST_CHUNK (64 * 1024)
#define TSIMD_ARENA_MAX_CHUNK   (4 * 1024 * 1024)

static unsigned char *chunk_data(tsimd_arena_chunk_t *c) {
    uintptr_t base = (uintptr_t)c + sizeof(*c);
    return (unsigned char *)((base + 63) & ~(uintptr_t)63);
}

int tsimd_arena_init(tsimd_arena_t *a) {
    a->head = NULL;
    a->next_chunk_size = TSIMD_ARENA_FIRST_CHUNK;
    return 0;
}

void tsimd_arena_reset(tsimd_arena_t *a) {
    tsimd_arena_chunk_t *c = a->head;
    while (c) {
        tsimd_arena_chunk_t *next = c->next;
        free(c);
        c = next;
    }
    a->head = NULL;
    a->next_chunk_size = TSIMD_ARENA_FIRST_CHUNK;
}

void *tsimd_arena_alloc_aligned(tsimd_arena_t *a, size_t size, size_t align) {
    if (size == 0) size = 1;
    for (tsimd_arena_chunk_t *c = a->head; c; c = c->next) {
        unsigned char *data = chunk_data(c);
        uintptr_t base = (uintptr_t)(data + c->used);
        uintptr_t aligned = (base + (align - 1)) & ~(uintptr_t)(align - 1);
        if (aligned + size <= (uintptr_t)(data + c->cap)) {
            c->used = (size_t)(aligned + size - (uintptr_t)data);
            return (void *)aligned;
        }
    }
    size_t cap = a->next_chunk_size;
    if (cap < size + align + 64) cap = size + align + 64;
    if (a->next_chunk_size < TSIMD_ARENA_MAX_CHUNK)
        a->next_chunk_size *= 2;
    tsimd_arena_chunk_t *c = malloc(sizeof(*c) + cap + 64);
    if (!c) {
        fprintf(stderr, "tsimd: out of memory (arena chunk %zu bytes)\n", cap);
        return NULL;
    }
    c->next = a->head;
    c->used = 0;
    c->cap = cap;
    a->head = c;
    unsigned char *data = chunk_data(c);
    uintptr_t base = (uintptr_t)data;
    uintptr_t aligned = (base + (align - 1)) & ~(uintptr_t)(align - 1);
    c->used = (size_t)(aligned + size - base);
    return (void *)aligned;
}

void *tsimd_arena_alloc(tsimd_arena_t *a, size_t size) {
    return tsimd_arena_alloc_aligned(a, size, sizeof(void *));
}

void *tsimd_xmalloc(size_t size) {
    void *p = malloc(size ? size : 1);
    if (!p) {
        fprintf(stderr, "tsimd: out of memory (%zu bytes)\n", size);
        abort();
    }
    return p;
}

void *tsimd_xmalloc_aligned(size_t size, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align,
                       size ? size : 1) != 0) {
        fprintf(stderr, "tsimd: out of memory (%zu bytes, align %zu)\n", size, align);
        abort();
    }
    return p;
}

void *tsimd_xrealloc(void *p, size_t size) {
    void *q = realloc(p, size ? size : 1);
    if (!q) {
        fprintf(stderr, "tsimd: out of memory (%zu bytes)\n", size);
        abort();
    }
    return q;
}

void tsimd_xfree(void *p) { free(p); }

#define TSIMD_HUGE_ALIGN ((size_t)2 << 20)

void *tsimd_alloc_huge(size_t bytes) {
    if (bytes == 0) bytes = 1;
    size_t rounded =
        (bytes + TSIMD_HUGE_ALIGN - 1) & ~(TSIMD_HUGE_ALIGN - 1);
    if (rounded >= TSIMD_HUGE_ALIGN) {
        void *p = aligned_alloc(TSIMD_HUGE_ALIGN, rounded);
        if (p) {
#if defined(MADV_HUGEPAGE)
            madvise(p, rounded, MADV_HUGEPAGE);
#endif
            return p;
        }
    }
    return malloc(bytes);
}

void *tsimd_alloc_huge_zero(size_t bytes) {
    if (bytes == 0) bytes = 1;
    size_t rounded =
        (bytes + TSIMD_HUGE_ALIGN - 1) & ~(TSIMD_HUGE_ALIGN - 1);
    if (rounded >= TSIMD_HUGE_ALIGN) {
        void *p = aligned_alloc(TSIMD_HUGE_ALIGN, rounded);
        if (p) {
#if defined(MADV_HUGEPAGE)
            madvise(p, rounded, MADV_HUGEPAGE);
#endif
            memset(p, 0, bytes);
            return p;
        }
    }
    return calloc(bytes, 1);
}
