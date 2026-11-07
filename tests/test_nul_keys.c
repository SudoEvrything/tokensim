#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const char *js =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{"
        "\"a\\u0000b\":0,\"xy\":1},\"merges\":[]},"
        "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"use_regex\":true,"
        "\"add_prefix_space\":false},\"added_tokens\":[]}";
    char path[] = "/tmp/tsimd_nulkeys_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 2;
    size_t jlen = strlen(js);
    if (write(fd, js, jlen) != (ssize_t)jlen) {
        close(fd);
        return 2;
    }
    close(fd);

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, path,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        unlink(path);
        return 2;
    }
    const uint8_t in[] = {'a', 0x00, 'b'};
    uint32_t pid = tsimd_vocab_find(&t.vocab, in, sizeof(in));
    int rc = 0;
    if (pid != 0) {
        fprintf(stderr,
                "FAIL: nul-embedded vocab key truncated: find=%u want 0\n",
                pid);
        rc = 1;
    }
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(&t, in, sizeof(in), &seq) != 0) {
        fprintf(stderr, "encode failed\n");
        rc = 1;
    } else if (seq.count == 0) {
        fprintf(stderr, "FAIL: empty encode\n");
        rc = 1;
    }
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    unlink(path);
    if (rc) return 1;
    printf("test_nul_keys: OK\n");
    return 0;
}
