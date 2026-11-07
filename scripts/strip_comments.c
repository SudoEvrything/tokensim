#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    *len = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[*len] = 0;
    return buf;
}

static void emit_stripped_c(const char *in, size_t n, char **out,
                            size_t *olen) {
    size_t cap = n + 16;
    char *o = malloc(cap);
    size_t j = 0;
    enum { CODE, LINE, BLOCK, STR, CHR } st = CODE;
    for (size_t i = 0; i < n; i++) {
        char c = in[i];
        char d = i + 1 < n ? in[i + 1] : 0;
        switch (st) {
            case CODE:
                if (c == '/' && d == '/') { st = LINE; i++; }
                else if (c == '/' && d == '*') { st = BLOCK; i++; if (j && o[j-1] != '\n' && j < cap) o[j++] = ' '; }
                else if (c == '"') { st = STR; if (j < cap) o[j++] = c; }
                else if (c == '\'') { st = CHR; if (j < cap) o[j++] = c; }
                else { if (j + 1 >= cap) { cap *= 2; o = realloc(o, cap); } o[j++] = c; }
                break;
            case LINE:
                if (c == '\n') { st = CODE; if (j < cap) o[j++] = c; }
                break;
            case BLOCK:
                if (c == '*' && d == '/') { st = CODE; i++; }
                break;
            case STR:
                if (c == '\\') { if (j + 2 >= cap) { cap *= 2; o = realloc(o, cap); } o[j++] = c; i++; if (i < n && j < cap) o[j++] = in[i]; }
                else if (c == '"') { st = CODE; if (j < cap) o[j++] = c; }
                else { if (j + 1 >= cap) { cap *= 2; o = realloc(o, cap); } o[j++] = c; }
                break;
            case CHR:
                if (c == '\\') { if (j + 2 >= cap) { cap *= 2; o = realloc(o, cap); } o[j++] = c; i++; if (i < n && j < cap) o[j++] = in[i]; }
                else if (c == '\'') { st = CODE; if (j < cap) o[j++] = c; }
                else { if (j + 1 >= cap) { cap *= 2; o = realloc(o, cap); } o[j++] = c; }
                break;
        }
    }
    *out = o;
    *olen = j;
}

static void collapse_blanks(char *o, size_t *j) {
    size_t w = 0, r = 0;
    int line_start = 1, blanks = 0;
    while (r < *j) {
        char c = o[r];
        if (c == '\n') {
            size_t e = w;
            while (e > 0 && (o[e-1] == ' ' || o[e-1] == '\t') &&
                   (e - 1 == 0 || o[e-2] != '\n')) {
                if (w > 0 && (o[w-1] == ' ' || o[w-1] == '\t')) w--;
                else break;
                e--;
            }
            r++;
            if (w == 0 || o[w-1] == '\n') {
                blanks++;
                if (blanks > 1) continue;
                if (w == 0) continue;
            } else {
                blanks = 0;
            }
            if (w < (*j)) o[w++] = '\n';
            line_start = 1;
            continue;
        }
        if (c == ' ' || c == '\t') {
            size_t k = r;
            while (k < *j && (o[k] == ' ' || o[k] == '\t')) k++;
            if (k < *j && o[k] == '\n') { r = k; continue; }
        }
        blanks = 0;
        o[w++] = c;
        r++;
    }
    while (w > 0 && (o[w-1] == '\n' || o[w-1] == ' ')) w--;
    o[w++] = '\n';
    *j = w;
}

static int strip_line_hash(const char *in, size_t n, char **out,
                           size_t *olen) {
    char *o = malloc(n + 16);
    size_t j = 0;
    size_t i = 0;
    int first = 1;
    while (i < n) {
        size_t e = i;
        while (e < n && in[e] != '\n') e++;
        size_t le = e;
        if (le > i && in[le-1] == '\r') le--;
        size_t k = i;
        while (k < le && (in[k] == ' ' || in[k] == '\t')) k++;
        int is_comment = (k < le && in[k] == '#');
        int is_shebang = first && k == i && k + 1 < le && in[k+1] == '!';
        if (!is_comment || is_shebang) {
            memcpy(o + j, in + i, le - i);
            j += le - i;
        }
        if (e < n) o[j++] = '\n';
        i = e < n ? e + 1 : n;
        first = 0;
    }
    *out = o;
    *olen = j;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <c|hash> <file>\n", argv[0]);
        return 2;
    }
    size_t n = 0;
    char *in = slurp(argv[2], &n);
    if (!in) { fprintf(stderr, "read %s failed\n", argv[2]); return 1; }
    char *mid = NULL;
    size_t mn = 0;
    if (strcmp(argv[1], "c") == 0) {
        emit_stripped_c(in, n, &mid, &mn);
        free(in);
    } else {
        strip_line_hash(in, n, &mid, &mn);
        free(in);
    }
    collapse_blanks(mid, &mn);
    FILE *f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "write %s failed\n", argv[2]); return 1; }
    fwrite(mid, 1, mn, f);
    fclose(f);
    free(mid);
    return 0;
}
