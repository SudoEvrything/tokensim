#include "input.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "allocator.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#define TSIMD_HAVE_MMAP 1
#endif

int tsimd_input_read_file(const char *path, tsimd_input_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return -1;
    }
    size_t size = (size_t)st.st_size;

    uint8_t *buf = tsimd_xmalloc(size + 1);
    size_t got = 0;
    while (got < size) {
        ssize_t n = read(fd, buf + got, size - got);
        if (n < 0) {
            free(buf);
            close(fd);
            return -1;
        }
        if (n == 0) break;
        got += (size_t)n;
    }
    close(fd);
    buf[got] = 0;

    out->data = buf;
    out->length = got;
    out->owned = buf;
    out->is_mmap = 0;
    return 0;
}

int tsimd_input_mmap_file(const char *path, tsimd_input_t *out) {
#if defined(TSIMD_HAVE_MMAP)
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return -1;
    }
    size_t size = (size_t)st.st_size;

    int flags = MAP_PRIVATE;
#if defined(MAP_POPULATE)
    {
        const char *e = getenv("TSIMD_NO_POPULATE");
        if (!(e && e[0] == '1') && size >= (64u << 20)) flags |= MAP_POPULATE;
    }
#endif
    void *map = mmap(NULL, size, PROT_READ, flags, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return -1;

    madvise(map, size, MADV_SEQUENTIAL);

    {
        const char *e = getenv("TSIMD_NO_WILLNEED");
        if (!(e && e[0] == '1')) madvise(map, size, MADV_WILLNEED);
    }
#if defined(MADV_HUGEPAGE)
    if (size >= (2u << 20)) madvise(map, size, MADV_HUGEPAGE);
#endif

    out->data = map;
    out->length = size;
    out->owned = map;
    out->is_mmap = 1;
    return 0;
#else
    (void)path;
    (void)out;
    return -1;
#endif
}

int tsimd_input_from_buffer(const uint8_t *data, size_t length, tsimd_input_t *out) {
    out->data = data;
    out->length = length;
    out->owned = NULL;
    out->is_mmap = 0;
    return 0;
}

void tsimd_input_release(tsimd_input_t *in) {
#if defined(TSIMD_HAVE_MMAP)
    if (in->is_mmap && in->owned) munmap(in->owned, in->length);
    else
#endif
    tsimd_xfree(in->owned);
    in->data = NULL;
    in->length = 0;
    in->owned = NULL;
    in->is_mmap = 0;
}
