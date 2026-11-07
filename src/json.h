#ifndef TSIMD_JSON_H
#define TSIMD_JSON_H

#include <stddef.h>
#include <stdint.h>

#include "allocator.h"

typedef enum {
    TSIMD_JSON_NULL = 0,
    TSIMD_JSON_BOOL,
    TSIMD_JSON_INT,
    TSIMD_JSON_REAL,
    TSIMD_JSON_STRING,
    TSIMD_JSON_ARRAY,
    TSIMD_JSON_OBJECT
} tsimd_json_type_t;

typedef struct tsimd_json_value tsimd_json_value_t;

struct tsimd_json_value {
    uint8_t type;
    union {
        int boolean;
        long long i;
        double d;
        struct {
            const char *ptr;
            uint32_t len;
        } str;
        struct {
            const tsimd_json_value_t **items;
            size_t n;
        } arr;
        struct {
            const char **keys;
            uint32_t *klens;
            const tsimd_json_value_t **vals;
            size_t n;
        } obj;
    } u;
};

const tsimd_json_value_t *tsimd_json_parse(tsimd_arena_t *arena,
                                         const char *buf, size_t len,
                                         size_t *err_offset);

const tsimd_json_value_t *tsimd_json_obj_get(const tsimd_json_value_t *v,
                                           const char *key);
const char *tsimd_json_string(const tsimd_json_value_t *v);
long long tsimd_json_int(const tsimd_json_value_t *v);
int tsimd_json_is_null(const tsimd_json_value_t *v);

#endif
