#include "unicode.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

int main(void) {

    TEST_ASSERT(tsimd_class_ascii('a') == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_ascii('Z') == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_ascii('5') == TSIMD_CLASS_NUMBER);
    TEST_ASSERT(tsimd_class_ascii(' ') == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_ascii('\t') == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_ascii('\n') == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_ascii('\r') == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_ascii('!') == TSIMD_CLASS_OTHER);

    TEST_ASSERT(tsimd_class_ascii(0x1C) == TSIMD_CLASS_OTHER);
    TEST_ASSERT(tsimd_class_ascii(0x7F) == TSIMD_CLASS_OTHER);
    TEST_ASSERT(tsimd_class_ascii(0x00) == TSIMD_CLASS_OTHER);

    uint32_t cp;
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"h", 1, &cp), 1);
    TEST_ASSERT_EQ(cp, 'h');
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xC3\xA9", 2, &cp), 2);
    TEST_ASSERT_EQ(cp, 0xE9);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xE4\xB8\xAD", 3, &cp), 3);
    TEST_ASSERT_EQ(cp, 0x4E2D);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xF0\x9F\x9A\x80", 4, &cp), 4);
    TEST_ASSERT_EQ(cp, 0x1F680);

    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xC0\x80", 2, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xE0\x80\x80", 3, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xED\xA0\x80", 3, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xE4\xB8", 2, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xE4\xB8\x41", 3, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\x80", 1, &cp), 1);
    TEST_ASSERT_EQ(tsimd_utf8_decode((const uint8_t *)"\xF8\x80\x80\x80\x80", 5, &cp), 1);

    TEST_ASSERT(tsimd_class_codepoint(0xE9) == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_codepoint(0x4E2D) == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_codepoint(0x0627) == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_codepoint(0x0928) == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_codepoint(0x041F) == TSIMD_CLASS_LETTER);
    TEST_ASSERT(tsimd_class_codepoint(0x00BD) == TSIMD_CLASS_NUMBER);
    TEST_ASSERT(tsimd_class_codepoint(0x0660) == TSIMD_CLASS_NUMBER);
    TEST_ASSERT(tsimd_class_codepoint(0x00A0) == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_codepoint(0x3000) == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_codepoint(0x2028) == TSIMD_CLASS_SPACE);
    TEST_ASSERT(tsimd_class_codepoint(0x1F680) == TSIMD_CLASS_OTHER);
    TEST_ASSERT(tsimd_class_codepoint(0x0301) == TSIMD_CLASS_OTHER);
    TEST_ASSERT(tsimd_class_codepoint(0x200B) == TSIMD_CLASS_OTHER);

    {
        tsimd_char_class_t cls;
        uint32_t c2;
        static const char *inv[] = {
            "\x80", "\xa0", "\xbf", "\xc3 ", "\xd7 ", "\xe3\x41",
            "\xc2", "\xf0\x9f\x28",
        };
        for (size_t i = 0; i < sizeof(inv) / sizeof(inv[0]); i++) {
            size_t n = strlen(inv[i]);
            size_t adv = tsimd_utf8_next((const uint8_t *)inv[i], n, &c2,
                                         &cls);
            TEST_ASSERT_EQ(adv, 1);
            TEST_ASSERT(cls == TSIMD_CLASS_OTHER);
        }
        TEST_ASSERT(tsimd_utf8_next((const uint8_t *)"\xc2\xa0", 2, &c2,
                                    &cls) == 2);
        TEST_ASSERT(cls == TSIMD_CLASS_SPACE);
        TEST_ASSERT(tsimd_utf8_next((const uint8_t *)"\xc2\xbd", 2, &c2,
                                    &cls) == 2);
        TEST_ASSERT(cls == TSIMD_CLASS_NUMBER);
        TEST_ASSERT(tsimd_utf8_next((const uint8_t *)" ", 1, &c2, &cls) == 1);
        TEST_ASSERT(cls == TSIMD_CLASS_SPACE);
        TEST_ASSERT(tsimd_utf8_next((const uint8_t *)"a", 1, &c2, &cls) == 1);
        TEST_ASSERT(cls == TSIMD_CLASS_LETTER);
        TEST_ASSERT(tsimd_utf8_next((const uint8_t *)"!a", 1, &c2, &cls) == 1);
        TEST_ASSERT(cls == TSIMD_CLASS_OTHER);
    }

    printf("test_unicode: OK\n");
    return 0;
}
