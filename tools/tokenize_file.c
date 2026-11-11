














#include "tokenizer.h"
#include "input.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *special_list = NULL;
    const char *format = "tiktoken";
    int n_threads = 0;
    int use_mmap = 1;
    int argi = 1;
    while (argi < argc && strncmp(argv[argi], "--", 2) == 0) {
        const char *opt = argv[argi];
        const char *val = NULL;
        char optbuf[32];
        const char *eq = strchr(opt, '=');
        if (eq) {
            if ((size_t)(eq - opt) >= sizeof(optbuf)) {
                fprintf(stderr, "unknown option %s\n", opt);
                return 2;
            }
            memcpy(optbuf, opt, (size_t)(eq - opt));
            optbuf[eq - opt] = '\0';
            opt = optbuf;
            val = eq + 1;
            argi += 1;
        }

        #define NEXT_VAL() (val ? val : (argi + 1 < argc ? argv[++argi] : NULL))
        if (strcmp(opt, "--special") == 0) {
            val = NEXT_VAL();
            if (!val) {
                fprintf(stderr, "missing argument after --special\n");
                return 2;
            }
            special_list = val;
            if (!eq) argi += 1;
        } else if (strcmp(opt, "--format") == 0) {
            val = NEXT_VAL();
            if (!val) {
                fprintf(stderr, "missing argument after --format\n");
                return 2;
            }
            format = val;
            if (!eq) argi += 1;
        } else if (strcmp(opt, "--threads") == 0) {
            val = NEXT_VAL();
            if (!val) {
                fprintf(stderr, "missing argument after --threads\n");
                return 2;
            }
            char *end = NULL;
            long v = strtol(val, &end, 10);
            if (end == val || (end && *end != '\0') || v < 0 ||
                v > 1024) {
                fprintf(stderr, "invalid --threads value: %s\n", val);
                return 2;
            }
            n_threads = (int)v;
            if (!eq) argi += 1;
        } else if (strcmp(opt, "--no-mmap") == 0) {
            use_mmap = 0;
            argi += 1;
        } else {
            fprintf(stderr, "unknown option %s\n", argv[argi]);
            return 2;
        }
        #undef NEXT_VAL
    }

    if (argc - argi < 2 || argc - argi > 4) {
        fprintf(stderr,
                "usage: %s [--format tiktoken|gpt2|tokenizer_json|sentencepiece] "
                "[--special tok1,tok2,...] MODEL [MODEL2] (gpt2|cl100k|auto) [FILE]\n",
                argv[0]);
        return 2;
    }
    const char *path1 = argv[argi++];
    const char *path2 = NULL;
    if (strcmp(format, "gpt2") == 0 && argc - argi >= 1 &&
        strcmp(argv[argi], "gpt2") != 0 && strcmp(argv[argi], "cl100k") != 0 &&
        strcmp(argv[argi], "o200k") != 0 && strcmp(argv[argi], "auto") != 0) {
        path2 = argv[argi++];
    }

    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    int auto_scheme = 0;
    if (argc - argi >= 1) {
        if (strcmp(argv[argi], "gpt2") == 0) {
            scheme = TSIMD_PRETOK_GPT2;
        } else if (strcmp(argv[argi], "cl100k") == 0) {
            scheme = TSIMD_PRETOK_CL100K;
        } else if (strcmp(argv[argi], "o200k") == 0) {
            scheme = TSIMD_PRETOK_O200K;
        } else if (strcmp(argv[argi], "spm") == 0) {
            scheme = TSIMD_PRETOK_SPM;
        } else if (strcmp(argv[argi], "kimi") == 0) {
            scheme = TSIMD_PRETOK_KIMI;
        } else if (strcmp(argv[argi], "gem4") == 0) {
            scheme = TSIMD_PRETOK_GEM4;
        } else if (strcmp(argv[argi], "deepseek") == 0) {
            scheme = TSIMD_PRETOK_DEEPSEEK;
        } else if (strcmp(argv[argi], "nemo") == 0) {
            scheme = TSIMD_PRETOK_NEMO;
        } else if (strcmp(argv[argi], "qwen25") == 0) {
            scheme = TSIMD_PRETOK_QWEN25;
        } else if (strcmp(argv[argi], "auto") == 0) {
            scheme = TSIMD_PRETOK_GPT2;
            auto_scheme = 1;
        } else {
            fprintf(stderr, "unknown scheme: %s (want gpt2|cl100k|o200k|spm|kimi|gem4|deepseek|nemo|qwen25|auto)\n",
                    argv[argi]);
            return 2;
        }
        argi++;
    } else {
        fprintf(stderr, "missing scheme argument\n");
        return 2;
    }

    tsimd_model_format_t fmt;
    if (strcmp(format, "tiktoken") == 0) fmt = TSIMD_MODEL_TIKTOKEN;
    else if (strcmp(format, "gpt2") == 0) fmt = TSIMD_MODEL_GPT2;
    else if (strcmp(format, "tokenizer_json") == 0) fmt = TSIMD_MODEL_TOKENIZER_JSON;
    else if (strcmp(format, "sentencepiece") == 0) fmt = TSIMD_MODEL_SENTENCEPIECE;
    else {
        fprintf(stderr, "unknown format: %s\n", format);
        return 2;
    }
    if (auto_scheme && fmt == TSIMD_MODEL_TIKTOKEN) {



        fprintf(stderr,
                "--scheme auto cannot detect the pretokenizer of a .tiktoken\n"
                "file (it holds mergeable ranks only). Pass an explicit\n"
                "--scheme gpt2|cl100k|o200k.\n");
        return 2;
    }
    if (fmt == TSIMD_MODEL_SENTENCEPIECE) {








        if (special_list) {
            fprintf(stderr,
                    "--special with sentencepiece is unsupported (the\n"
                    "reference ids for interspersed specials are not\n"
                    "part of the parity gates); pass no specials\n");
            return 2;
        }
    }

    tsimd_added_token_t specials[64] = {{0}};
    size_t n_specials = 0;
    char *copy = NULL;
    if (special_list) {
        copy = strdup(special_list);
        char *p = strtok(copy, ",");
        while (p) {
            if (n_specials >= 64) {
                fprintf(stderr,
                        "too many --special entries (max 64)\n");
                free(copy);
                return 2;
            }
            specials[n_specials].text = p;
            specials[n_specials].length = (uint32_t)strlen(p);
            specials[n_specials].is_special = 1;
            n_specials++;
            p = strtok(NULL, ",");
        }
    }

    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load_model(&t, fmt, path1, path2, &scheme, specials,
                                  n_specials) != 0) {
        fprintf(stderr, "failed to load model\n");
        free(copy);
        return 1;
    }

    tsimd_input_t in;
    if (argc - argi == 1) {
        if (use_mmap && tsimd_input_mmap_file(argv[argi], &in) == 0) {

        } else if (tsimd_input_read_file(argv[argi], &in) != 0) {
            fprintf(stderr, "failed to read: %s\n", argv[argi]);
            return 1;
        }
    } else {

        size_t cap = 1 << 20, len = 0;
        uint8_t *buf = malloc(cap);
        if (!buf) {
            fprintf(stderr, "out of memory\n");
            tsimd_tokenizer_free(&t);
            free(copy);
            return 1;
        }
        for (;;) {
            if (len == cap) {
                cap *= 2;
                uint8_t *grown = realloc(buf, cap);
                if (!grown) {
                    fprintf(stderr, "out of memory\n");
                    free(buf);
                    tsimd_tokenizer_free(&t);
                    free(copy);
                    return 1;
                }
                buf = grown;
            }
            size_t got = fread(buf + len, 1, cap - len, stdin);
            if (got == 0) {
                if (ferror(stdin)) {
                    fprintf(stderr, "read error on stdin\n");
                    free(buf);
                    return 1;
                }
                break;
            }
            len += got;
        }
        in.data = buf;
        in.length = len;
        in.owned = buf;
    }

    tsimd_token_sequence_t seq;
    if (n_threads != 1) {
        if (tsimd_tokenizer_encode_parallel(&t, in.data, in.length,
                                           n_threads, &seq) != 0) {
            fprintf(stderr, "tokenization failed\n");
            tsimd_tokenizer_free(&t);
            tsimd_input_release(&in);
            free(copy);
            return 1;
        }
    } else if (tsimd_tokenizer_encode(&t, in.data, in.length, &seq) != 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    if (getenv("TSIMD_STATS")) {
        size_t hits, misses, inserts;
        tsimd_tokenizer_cache_stats(&t, &hits, &misses, &inserts);
        double total = (double)(hits + misses);
        fprintf(stderr,
                "pretok cache: %zu inserts, %zu hits, %zu misses (%.1f%% hit)\n",
                inserts, hits, misses,
                total > 0.0 ? 100.0 * (double)hits / total : 0.0);
    }


    {
        char *buf = malloc(seq.count * 12 + 2);
        if (!buf) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        char *p = buf;
        for (size_t i = 0; i < seq.count; i++) {
            uint32_t v = seq.ids[i];
            if (i) *p++ = ' ';
            char tmp[10];
            int n = 0;
            do {
                tmp[n++] = (char)('0' + v % 10);
                v /= 10;
            } while (v);
            while (n) *p++ = tmp[--n];
        }
        *p++ = '\n';
        fwrite(buf, 1, (size_t)(p - buf), stdout);
        free(buf);
    }
    fprintf(stderr, "%zu bytes -> %zu tokens\n", in.length, seq.count);

    tsimd_token_sequence_free(&seq);
    tsimd_input_release(&in);
    tsimd_tokenizer_free(&t);
    free(copy);
    return 0;
}
