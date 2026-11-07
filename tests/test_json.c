#include "../src/json.h"
#include "util.h"

#include <string.h>

static int str_eq(const tsimd_json_value_t *v, const char *want) {
    const char *s = tsimd_json_string(v);
    return s && strcmp(s, want) == 0;
}

int main(void) {
    tsimd_arena_t arena;

    {
        const char *doc =
            "{\"a\": 1, \"b\": -2, \"s\": \"he\\u0120llo \\uD83D\\uDE00\\n\","
            " \"f\": 3.5, \"t\": true, \"n\": null,"
            " \"arr\": [1, \"two\", [3, {\"x\": 4}]]}";
        tsimd_arena_init(&arena);
        const tsimd_json_value_t *root =
            tsimd_json_parse(&arena, doc, strlen(doc), NULL);
        TEST_ASSERT(root != NULL);
        TEST_ASSERT(str_eq(tsimd_json_obj_get(root, "s"),
                           "he\xC4\xA0llo \xF0\x9F\x98\x80\n"));
        TEST_ASSERT_EQ(tsimd_json_int(tsimd_json_obj_get(root, "a")), 1);
        TEST_ASSERT_EQ(tsimd_json_int(tsimd_json_obj_get(root, "b")), -2);
        const tsimd_json_value_t *arr = tsimd_json_obj_get(root, "arr");
        TEST_ASSERT(arr && arr->type == TSIMD_JSON_ARRAY);
        TEST_ASSERT_EQ((long long)arr->u.arr.n, 3);
        TEST_ASSERT(str_eq(arr->u.arr.items[1], "two"));
        const tsimd_json_value_t *inner = arr->u.arr.items[2];
        TEST_ASSERT(inner && inner->type == TSIMD_JSON_ARRAY);
        TEST_ASSERT_EQ(tsimd_json_int(
                           tsimd_json_obj_get(inner->u.arr.items[1], "x")), 4);
        tsimd_arena_reset(&arena);
    }

    {
        const char *doc = "[50257, 100277, 0, 1e3]";
        tsimd_arena_init(&arena);
        const tsimd_json_value_t *root =
            tsimd_json_parse(&arena, doc, strlen(doc), NULL);
        TEST_ASSERT(root != NULL);
        TEST_ASSERT_EQ(tsimd_json_int(root->u.arr.items[0]), 50257);
        TEST_ASSERT_EQ(tsimd_json_int(root->u.arr.items[1]), 100277);
        TEST_ASSERT_EQ(tsimd_json_int(root->u.arr.items[2]), 0);
        TEST_ASSERT_EQ(tsimd_json_int(root->u.arr.items[3]), 1000);
        tsimd_arena_reset(&arena);
    }

    {
        const char *doc = "{\"k\": 1, \"k\": 2}";
        tsimd_arena_init(&arena);
        const tsimd_json_value_t *root =
            tsimd_json_parse(&arena, doc, strlen(doc), NULL);
        TEST_ASSERT(root != NULL);
        TEST_ASSERT_EQ(tsimd_json_int(tsimd_json_obj_get(root, "k")), 2);
        tsimd_arena_reset(&arena);
    }

    {
        const char *bad[] = {
            "", "{", "[1,", "{\"a\" 1}", "{\"a\": trxx}", "nul",
            "[1] trailing", "\"unterminated", "{\"a\": \"\\q\"}",
        };
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            tsimd_arena_init(&arena);
            size_t err = 0;
            const tsimd_json_value_t *root =
                tsimd_json_parse(&arena, bad[i], strlen(bad[i]), &err);
            TEST_ASSERT(root == NULL);
            tsimd_arena_reset(&arena);
        }

        char deep[600];
        size_t o = 0;
        for (int i = 0; i < 300; i++) deep[o++] = '[';
        deep[o] = 0;
        tsimd_arena_init(&arena);
        TEST_ASSERT(tsimd_json_parse(&arena, deep, o, NULL) == NULL);
        tsimd_arena_reset(&arena);
    }

    {
        const char *doc = "\"\\u00e9\"";
        tsimd_arena_init(&arena);
        const tsimd_json_value_t *root =
            tsimd_json_parse(&arena, doc, strlen(doc), NULL);
        TEST_ASSERT(root != NULL);
        TEST_ASSERT(str_eq(root, "\xC3\xA9"));
        tsimd_arena_reset(&arena);
    }

    printf("test_json: OK\n");
    return 0;
}
