#ifndef TSIMD_ALLOCATOR_H
#define TSIMD_ALLOCATOR_H

#include <stddef.h>

typedef struct tsimd_arena_chunk tsimd_arena_chunk_t;

typedef struct {
    tsimd_arena_chunk_t *head;
    size_t next_chunk_size;
} tsimd_arena_t;

int tsimd_arena_init(tsimd_arena_t *a);
void tsimd_arena_reset(tsimd_arena_t *a);
void *tsimd_arena_alloc(tsimd_arena_t *a, size_t size);

void *tsimd_arena_alloc_aligned(tsimd_arena_t *a, size_t size, size_t align);

void *tsimd_xmalloc(size_t size);
void *tsimd_xmalloc_aligned(size_t size, size_t align);
void *tsimd_xrealloc(void *p, size_t size);
void tsimd_xfree(void *p);

void *tsimd_alloc_huge(size_t bytes);
void *tsimd_alloc_huge_zero(size_t bytes);

#endif
