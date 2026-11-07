#include "trie.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static uint32_t walk_step(const tsimd_trie_t *t, uint32_t cur, uint8_t b,
                          uint32_t *idx) {
    return tsimd_trie_step(t, cur, b, idx);
}

static uint32_t walk_no_child(const tsimd_trie_t *t, uint32_t cur, uint8_t b) {
    uint32_t cu = t->u[cur];
    uint32_t idx = (cu >> TSIMD_TRIE_BASE_SHIFT) + b;
    if ((t->u[idx] & TSIMD_TRIE_LABEL) != b) return TSIMD_TRIE_NIL;
    return idx;
}

static void walk_expect(const tsimd_trie_t *t, const uint8_t *key, size_t len,
                        const uint32_t *ids, const char *tag) {
    uint32_t cur = tsimd_trie_root(t);
    for (size_t i = 0; i < len; i++) {
        uint32_t idx = 0;
        cur = walk_step(t, cur, key[i], &idx);
        if (cur == TSIMD_TRIE_NIL) {
            fprintf(stderr, "FAIL: %s: step %zu (byte 0x%02X) failed\n", tag, i,
                    key[i]);
            fails++;
            return;
        }
        uint32_t id = tsimd_trie_piece_id(t, cur, idx);
        if (id != ids[i]) {
            fprintf(stderr, "FAIL: %s: step %zu id %u expected %u\n", tag, i, id,
                    ids[i]);
            fails++;
        }
    }
}

int main(void) {
    uint32_t nil = TSIMD_TRIE_NIL;

    {
        static const uint8_t k0[] = {'a'};
        static const uint8_t k1[] = {'a', 'b'};
        static const uint8_t k2[] = {'a', 'b', 'c'};
        static const uint8_t k3[] = {'b'};
        static const uint8_t k4[] = {0x00};
        static const uint8_t k5[] = {0xC3, 0xA9};
        static const uint8_t k6[] = "0123456789abcdefghij";
        const uint8_t *keys[] = {k0, k1, k2, k3, k4, k5, k6};
        const uint32_t lens[] = {1, 2, 3, 1, 1, 2, 20};
        const uint32_t ids[] = {900, 901, 902, 903, 904, 905, 906};
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, 7);
        check(t != NULL, "build returns a trie");
        if (!t) return 1;

        uint32_t node = tsimd_trie_root(t);
        check(node != nil && (node & TSIMD_TRIE_PIECE) == 0,
              "the root is not a piece");

        {
            static const uint32_t want[] = {900, 901, 902};
            walk_expect(t, (const uint8_t *)"abc", 3, want, "abc");
        }
        {
            static const uint32_t want[] = {903};
            walk_expect(t, (const uint8_t *)"b", 1, want, "b");
        }
        {
            static const uint32_t want[] = {904};
            const uint8_t z = 0x00;
            walk_expect(t, &z, 1, want, "nul");
        }
        {
            static const uint8_t eacute[] = {0xC3, 0xA9};
            static const uint32_t want[] = {TSIMD_TRIE_NIL, 905};
            walk_expect(t, eacute, 2, want, "e-acute");
        }
        {
            static const uint8_t lng[] = "0123456789abcdefghij";
            static const uint32_t want[20] = {
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, TSIMD_TRIE_NIL, TSIMD_TRIE_NIL,
                TSIMD_TRIE_NIL, 906};
            walk_expect(t, lng, 20, want, "long");
        }

        uint32_t idx = 0;
        node = tsimd_trie_root(t);
        check(walk_step(t, node, (uint8_t)'z', &idx) == nil,
              "a missing first byte fails");
        node = walk_step(t, node, (uint8_t)'a', &idx);
        check(node != nil, "a steps");
        check(walk_step(t, node, (uint8_t)'z', &idx) == nil,
              "a missing second byte fails");
        node = walk_step(t, node, (uint8_t)'b', &idx);
        check(walk_step(t, node, (uint8_t)'c', &idx) != nil, "abc steps");
        node = walk_step(t, node, (uint8_t)'c', &idx);
        check(walk_step(t, node, (uint8_t)'d', &idx) == nil,
              "a piece is not a prefix of everything");
        node = walk_step(t, tsimd_trie_root(t), (uint8_t)'a', &idx);
        check(tsimd_trie_piece_id(t, node, idx) == 900,
              "the prefix piece keeps its id");
        node = walk_step(t, node, (uint8_t)'b', &idx);
        check(tsimd_trie_piece_id(t, node, idx) == 901, "ab is a piece");
        tsimd_trie_free(t);
    }

    {
        const uint8_t *keys[2] = {NULL, NULL};
        const uint32_t lens[2] = {0, 0};
        static const uint32_t eids[2] = {0, 1};
        tsimd_trie_t *et = tsimd_trie_build(keys, lens, eids, NULL, 2);
        check(et != NULL, "empty keys do not fail the build");
        check((tsimd_trie_root(et) & TSIMD_TRIE_PIECE) == 0,
              "an empty key leaves the root without an id");
        tsimd_trie_free(et);
        tsimd_trie_t *zt = tsimd_trie_build(NULL, NULL, NULL, NULL, 0);
        check(zt != NULL, "an empty build works");
        check((tsimd_trie_root(zt) & TSIMD_TRIE_CHILD) == 0,
              "the empty trie has no transitions");
        tsimd_trie_free(zt);
        tsimd_trie_free(NULL);
    }

    {
        static const uint8_t all[256][1] = {{0}};
        const uint8_t *keys[256];
        uint32_t lens[256];
        uint32_t ids[256];
        uint8_t buf[256][1];
        for (int i = 0; i < 256; i++) {
            buf[i][0] = (uint8_t)i;
            keys[i] = buf[i];
            lens[i] = 1;
            ids[i] = (uint32_t)(1000 + i);
        }
        (void)all;
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, 256);
        check(t != NULL, "the byte-complete build works");
        for (int i = 0; i < 256; i++) {
            uint32_t idx = 0;
            uint32_t node = walk_step(t, tsimd_trie_root(t), (uint8_t)i, &idx);
            if (node == nil ||
                tsimd_trie_piece_id(t, node, idx) != 1000u + (uint32_t)i) {
                fprintf(stderr, "FAIL: byte 0x%02X not found\n", i);
                fails++;
            }
        }
        tsimd_trie_free(t);
    }

    {
        static const uint8_t dup[] = {'x'};
        const uint8_t *keys[3] = {dup, dup, dup};
        const uint32_t lens[3] = {1, 1, 1};
        const uint32_t dids[3] = {50, 7, 99};
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, dids, NULL, 3);
        uint32_t idx = 0;
        uint32_t node = walk_step(t, tsimd_trie_root(t), (uint8_t)'x', &idx);
        check(tsimd_trie_piece_id(t, node, idx) == 7,
              "the lowest id wins a duplicate");
        tsimd_trie_free(t);
    }

    {
        enum { N = 4000 };
        uint8_t *store = malloc(N * 8);
        const uint8_t *keys[N];
        uint32_t lens[N];
        uint32_t ids[N];
        for (int i = 0; i < N; i++) {
            uint8_t *p = store + i * 8;
            p[0] = (uint8_t)(0x80 | (i >> 8));
            p[1] = (uint8_t)(i & 0xFF);
            p[2] = (uint8_t)(i >> 4);
            p[3] = (uint8_t)(i & 0x1F);
            keys[i] = p;
            lens[i] = 4;
            ids[i] = (uint32_t)(i * 3 + 1);
        }
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, N);
        check(t != NULL, "the 4000-key build works");
        int bad = 0;
        for (int i = 0; i < N; i++) {
            uint32_t node = tsimd_trie_root(t);
            uint32_t idx = 0;
            for (int k = 0; k < 4; k++)
                node = walk_step(t, node, keys[i][k], &idx);
            if (node == nil || tsimd_trie_piece_id(t, node, idx) != ids[i])
                bad++;
        }
        check(bad == 0, "every key of the 4000-key set resolves");
        tsimd_trie_free(t);
        free(store);
    }

    {
        enum { NS = 20000, WA = 8 };
        uint8_t *store = malloc(NS * 12);
        const uint8_t *keys[NS];
        uint32_t lens[NS];
        uint32_t ids[NS];
        unsigned seed = 12345u;
        for (int i = 0; i < NS; i++) {
            seed = seed * 1103515245u + 12345u;
            uint32_t len = 1 + (seed >> 16) % 12;
            uint8_t *p = store + i * 12;
            for (uint32_t k = 0; k < len; k++) {
                seed = seed * 1103515245u + 12345u;
                p[k] = (uint8_t)('a' + (seed >> 16) % WA);
            }
            keys[i] = p;
            lens[i] = len;
            ids[i] = (uint32_t)(i * 7 + 3);
        }
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, NS);
        check(t != NULL, "the stress build works");
        uint32_t *expect = malloc(NS * sizeof(uint32_t));
        for (int i = 0; i < NS; i++) {
            expect[i] = ids[i];
            for (int j = 0; j < i; j++) {
                if (lens[j] == lens[i] &&
                    memcmp(keys[j], keys[i], lens[i]) == 0) {
                    expect[i] = expect[j];
                    break;
                }
            }
        }
        int bad = 0, wrong = 0, checks = 0;
        for (int i = 0; i < NS; i++) {
            uint32_t cur = tsimd_trie_root(t);
            uint32_t idx = 0;
            for (uint32_t k = 0; k < lens[i]; k++) {
                cur = walk_step(t, cur, keys[i][k], &idx);
                if (cur == TSIMD_TRIE_NIL) {
                    bad++;
                    break;
                }
                (void)tsimd_trie_piece_id(t, cur, idx);
            }
            if (lens[i] > 0) {
                if (cur != TSIMD_TRIE_NIL &&
                    tsimd_trie_piece_id(t, cur, idx) != expect[i])
                    bad++;
            }
            checks++;
        }
        free(expect);
        check(bad == 0, "every stress key resolves");
        check(wrong == 0, "no stress prefix reports a foreign id");
        enum { NR = 20000 };
        for (int r = 0; r < NR; r++) {
            uint8_t pat[16];
            uint32_t len = 1 + (unsigned)r % 8;
            for (uint32_t k = 0; k < len; k++) {
                seed = seed * 1103515245u + 12345u;
                pat[k] = (uint8_t)('a' + (seed >> 16) % (WA + 3));
            }
            uint32_t cur = tsimd_trie_root(t);
            uint32_t idx = 0;
            for (uint32_t k = 0; k < len; k++) {
                cur = walk_step(t, cur, pat[k], &idx);
                if (cur == TSIMD_TRIE_NIL) break;
                uint32_t id = tsimd_trie_piece_id(t, cur, idx);
                if (id == TSIMD_TRIE_NIL) continue;
                int found = 0;
                for (int j = 0; j < NS; j++) {
                    if (lens[j] == k + 1 && ids[j] == id &&
                        memcmp(keys[j], pat, k + 1) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    if (wrong == 0)
                        fprintf(stderr, "FP: depth=%u id=%u\n", k + 1, id);
                    wrong++;
                }
            }
        }
        check(wrong == 0, "no random walk reports a foreign id");
        tsimd_trie_free(t);
        free(store);
    }

    {
        enum { NB = 20000 };
        uint8_t *store = malloc(NB * 10);
        const uint8_t *keys[NB];
        uint32_t lens[NB];
        uint32_t ids[NB];
        unsigned sd = 999u;
        for (int i = 0; i < NB; i++) {
            sd = sd * 1103515245u + 12345u;
            uint32_t len = 1 + (sd >> 16) % 10;
            uint8_t *p = store + i * 10;
            for (uint32_t k = 0; k < len; k++) {
                sd = sd * 1103515245u + 12345u;
                p[k] = (uint8_t)('A' + (sd >> 16) % 26);
            }
            keys[i] = p;
            lens[i] = len;
            ids[i] = (uint32_t)(i + 11);
        }
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, NB);
        check(t != NULL, "the invariant build works");
        if (t) {
            int shared = 0, bad = 0;
            for (uint32_t i = 0; i < t->n; i++) {
                if (!(t->u[i] & TSIMD_TRIE_CHILD)) continue;
                uint32_t base = tsimd_trie_base_of(t, i);
                for (uint32_t j = i + 1; j < t->n; j++) {
                    if ((t->u[j] & TSIMD_TRIE_CHILD) &&
                        tsimd_trie_base_of(t, j) == base)
                        shared++;
                }
            }
            check(shared == 0, "no two nodes share a base");
            int selflabel = 0;
            for (uint32_t i = 1; i < 256; i++)
                if ((t->u[i] & TSIMD_TRIE_LABEL) == i) selflabel++;
            check(selflabel == 0,
                  "no unit below 256 carries its own index as a label");
            int leaked = 0;
            for (uint32_t i = 0; i < t->n; i++) {
                if (t->u[i] & TSIMD_TRIE_CHILD) continue;
                for (int b = 0; b < 256; b++) {
                    uint32_t j = 0;
                    if (walk_step(t, i, (uint8_t)b, &j) != TSIMD_TRIE_NIL)
                        leaked++;
                }
            }
            check(leaked == 0, "no leaf reaches any unit");
            for (int i = 0; i < NB; i++) {
                uint32_t cur = tsimd_trie_root(t);
                uint32_t idx = 0;
                for (uint32_t k = 0; k < lens[i]; k++)
                    cur = walk_step(t, cur, keys[i][k], &idx);
                if (cur == TSIMD_TRIE_NIL ||
                    tsimd_trie_piece_id(t, cur, idx) == TSIMD_TRIE_NIL)
                    bad++;
            }
            check(bad == 0, "every invariant key resolves");
            tsimd_trie_free(t);
        }
        free(store);
    }

    {
        static const uint8_t q0[] = {'q'};
        static const uint8_t q1[] = {'q', 'r'};
        static const uint8_t q2[] = {'q', 'r', 's'};
        const uint8_t *keys[] = {q0, q1, q2};
        const uint32_t lens[] = {1, 2, 3};
        const uint32_t ids[] = {70, 71, 72};
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, 3);
        check(t != NULL, "the leaf-wall build works");
        if (t) {
            uint32_t idx = 0;
            uint32_t leaf = tsimd_trie_root(t);
            leaf = walk_step(t, leaf, (uint8_t)'q', &idx);
            leaf = walk_step(t, leaf, (uint8_t)'r', &idx);
            leaf = walk_step(t, leaf, (uint8_t)'s', &idx);
            check(leaf != nil && (t->u[leaf] & TSIMD_TRIE_CHILD) == 0,
                  "the deepest node of qrs is a leaf");
            check(tsimd_trie_base_of(t, leaf) == 0, "a leaf stores base 0");
            int refused = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t j = 0;
                if (walk_step(t, leaf, (uint8_t)b, &j) == nil) refused++;
            }
            check(refused == 256, "a leaf refuses all 256 bytes");
            check((t->u[0] & TSIMD_TRIE_LABEL) == 0,
                  "the root carries label 0");
            check(walk_no_child(t, leaf, 0) != nil,
                  "the label test alone accepts byte 0 from a leaf");
            check(walk_no_child(t, leaf, 0) == 0,
                  "the label test alone sends byte 0 from a leaf to the root");
            tsimd_trie_free(t);
        }
    }

    {
        static const uint8_t a0[] = {'a'};
        static const uint8_t a1[] = {'a', 'b'};
        static const uint8_t a2[] = {'a', 'b', 'c'};
        static const uint8_t a3[] = {'b'};
        const uint8_t *keys[] = {a0, a1, a2, a3};
        const uint32_t lens[] = {1, 2, 3, 1};
        const uint32_t ids[] = {10, 11, 12, 13};
        tsimd_trie_t *t = tsimd_trie_build(keys, lens, ids, NULL, 4);
        check(t != NULL, "the carried-step build works");
        if (t) {
            int mismatch = 0;
            for (uint32_t i = 0; i < t->n; i++) {
                for (int b = 0; b < 256; b++) {
                    uint32_t j1 = 0, j2 = 0, nu = 0;
                    uint32_t x = walk_step(t, i, (uint8_t)b, &j1);
                    uint32_t y =
                        tsimd_trie_advance(t, t->u[i], (uint8_t)b, &j2, &nu);
                    if ((x == nil) != (y == nil)) {
                        mismatch++;
                        continue;
                    }
                    if (x != nil && (x != y || j1 != j2 || nu != t->u[x]))
                        mismatch++;
                }
            }
            check(mismatch == 0,
                  "the carried step agrees with the plain step on every node");
            tsimd_trie_free(t);
        }
        tsimd_trie_t *et = tsimd_trie_build(NULL, NULL, NULL, NULL, 0);
        check(et != NULL, "the empty build works for the carried step");
        if (et) {
            uint32_t j = 0, nu = 0;
            check(tsimd_trie_advance(et, et->u[0], 0, &j, &nu) == nil,
                  "the carried step on an empty trie fails");
            check(tsimd_trie_advance(et, et->u[0], 0xFF, &j, &nu) == nil,
                  "the carried step on an empty trie fails for 0xFF");
            tsimd_trie_free(et);
        }
    }

    if (fails) {
        fprintf(stderr, "%d trie cases FAILED\n", fails);
        return 1;
    }
    printf("test_spm_trie: all cases pass\n");
    return 0;
}
