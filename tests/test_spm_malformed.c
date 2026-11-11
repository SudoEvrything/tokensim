#include "tokensimd_sys.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[] = "/tmp/tsimd_spm_malXXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    unsigned char crafted[] = {
        0x0A, 0x64,
        0x0A, 0xC8, 0x01,
        'a', 'b', 'c', 'd',
        0x15, 0x00, 0x00, 0xC8, 0x42,
        0x0A, 0x02, 'x', 'y',
        0x15, 0x00, 0x00, 0xC8, 0x42
    };
    if (write(fd, crafted, sizeof crafted) != (ssize_t)sizeof crafted) {
        close(fd);
        return 1;
    }
    close(fd);
    char err[256];
    tsimd_sys *h = tsimd_sys_open(path, "sentencepiece", "auto", err,
                                  sizeof err);
    unlink(path);
    if (h != NULL) {
        tsimd_sys_close(h);
        fprintf(stderr, "the malformed model loaded - must be rejected\n");
        return 1;
    }
    printf("test_spm_malformed: the truncated protobuf rejected cleanly "
           "(%s)\n", err);
    return 0;
}
