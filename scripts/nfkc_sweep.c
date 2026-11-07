#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicode.h"

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "r");
    if (!f) return 2;
    static uint8_t in[64], out[256];
    char line[512];
    while (fgets(line, sizeof line, f)) {
        size_t n = 0;
        char *p = line;
        while (*p && *p != '\n') {
            unsigned b;
            sscanf(p, "%2x", &b);
            in[n++] = (uint8_t)b;
            p += 2;
        }
        size_t on = tsimd_nfkc_normalize(in, n, out, sizeof out);
        if (on == (size_t)-1) {
            printf("ERR\n");
            continue;
        }
        for (size_t i = 0; i < on; i++) printf("%02x", out[i]);
        printf("\n");
    }
    fclose(f);
    return 0;
}
