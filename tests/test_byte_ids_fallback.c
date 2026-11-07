#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
static int checks = 0;

int main(void) {
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &tok, TSIMD_MODEL_TOKENIZER_JSON,
            TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json", NULL, &sch, NULL,
            0) != 0) {
        fprintf(stderr, "FAIL: deepseek load\n");
        return 1;
    }
    for (int b = 0; b < 256; b++) {
        uint8_t one = (uint8_t)b;
        uint32_t bid = tok.vocab.byte_ids[b];
        checks++;
        if (bid >= tok.vocab.vocab_size) {
            fprintf(stderr, "FAIL byte_ids[0x%02x]=%u out of range\n", b, bid);
            fails++;
            continue;
        }
        uint32_t len = tok.vocab.offsets[bid + 1] - tok.vocab.offsets[bid];
        if (len != 1 || tok.vocab.bytes[tok.vocab.offsets[bid]] != one) {
            fprintf(stderr,
                    "FAIL byte_ids[0x%02x]=%u points at the wrong text\n", b,
                    bid);
            fails++;
            continue;
        }
        uint32_t fid = tsimd_vocab_find(&tok.vocab, &one, 1);
        checks++;
        if (fid != bid) {
            fprintf(stderr, "FAIL byte 0x%02x: byte_ids=%u but find=%u\n", b,
                    bid, fid);
            fails++;
        }
        checks++;
        if (tok.vocab.byte_ids_hex[b] != 0xFFFFFFFFu &&
            tok.vocab.byte_ids_hex[b] >= tok.vocab.vocab_size) {
            fprintf(stderr, "FAIL byte_ids_hex[0x%02x] out of range\n", b);
            fails++;
        }
    }
    tsimd_tokenizer_free(&tok);

    tsimd_tokenizer_t g4;
    memset(&g4, 0, sizeof g4);
    sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &g4, TSIMD_MODEL_TOKENIZER_JSON,
            TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json",
            NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: gemma4 load\n");
        return 1;
    }
    {
        const int spot[4] = {0x9e, 0x0b, 0x1a, 0x5d};
        for (int k = 0; k < 4; k++) {
            uint8_t one = (uint8_t)spot[k];
            uint32_t bid = g4.vocab.byte_ids[one];
            uint32_t fid = tsimd_vocab_find(&g4.vocab, &one, 1);
            checks++;
            if (bid == 0xFFFFFFFFu || bid >= g4.vocab.vocab_size) {
                fprintf(stderr, "FAIL gemma4 byte_ids[0x%02x] unset\n", one);
                fails++;
                continue;
            }
            uint32_t len = g4.vocab.offsets[bid + 1] - g4.vocab.offsets[bid];
            checks++;
            if (len != 1 || g4.vocab.bytes[g4.vocab.offsets[bid]] != one) {
                fprintf(stderr, "FAIL gemma4 byte_ids[0x%02x] wrong text\n",
                        one);
                fails++;
            }
            checks++;
            if (fid != bid) {
                fprintf(stderr,
                        "FAIL gemma4 byte 0x%02x: byte_ids=%u find=%u\n", one,
                        bid, fid);
                fails++;
            }
        }
        checks++;
        if (g4.vocab.byte_ids[0x9e] != 396) {
            fprintf(stderr, "FAIL gemma4 byte_ids[0x9e]=%u want 396\n",
                    g4.vocab.byte_ids[0x9e]);
            fails++;
        }
    }
    checks++;
    {
        const uint8_t bos[] = {0x3c, 0x73, 0x3e};
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode(&g4, bos, 3, &seq) != 0) {
            fprintf(stderr, "FAIL: gemma4 <s> encode rc\n");
            fails++;
        } else {
            if (seq.count != 3 || seq.ids[0] != 236820 ||
                seq.ids[1] != 236751 || seq.ids[2] != 236813) {
                fprintf(stderr, "FAIL gemma4 <s> encode: %zu ids:", seq.count);
                for (size_t i = 0; i < seq.count && i < 4; i++)
                    fprintf(stderr, " %u", seq.ids[i]);
                fprintf(stderr, "\n");
                fails++;
            }
            tsimd_token_sequence_free(&seq);
        }
    }
    tsimd_tokenizer_free(&g4);
    printf("test_byte_ids_fallback: %d checks, %d fails\n", checks, fails);
    return fails != 0;
}
