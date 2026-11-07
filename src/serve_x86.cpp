#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

namespace {

struct Span {
    uint64_t k0;
    uint64_t k1;
    const uint8_t *ptr;
    uint64_t meta;
};
struct Entry {
    uint64_t key0;
    uint64_t key1;
    uint64_t val;
    uint64_t ext;
};
static_assert(sizeof(Span) == 32, "span layout changed");
static_assert(sizeof(Entry) == 32, "entry layout changed");
static_assert(offsetof(Span, meta) == 24, "span meta offset changed");
static_assert(offsetof(Entry, val) == 16, "entry val offset changed");

constexpr size_t kServeD = 12;

}

extern "C" size_t tsimd_serve_hot_avx2(const void *ents_v, size_t pair_mask,
                                       size_t mask, const void *batch_v,
                                       size_t n, size_t j0, uint32_t *out,
                                       size_t *cur_inout) {
    const Entry *ents = static_cast<const Entry *>(ents_v);
    const Span *batch = static_cast<const Span *>(batch_v);
    size_t cur = *cur_inout;
    size_t j = j0;

    for (; j < n; j++) {
        if (j + 1 < n) {
            const __m128i wantA = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(&batch[j].k0));
            const __m128i wantB = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(&batch[j + 1].k0));
            const uint64_t k1a =
                static_cast<uint64_t>(_mm_extract_epi64(wantA, 1));
            const uint64_t k1b =
                static_cast<uint64_t>(_mm_extract_epi64(wantB, 1));
            if ((k1a >> 56) != 0 && (k1b >> 56) != 0) {
                const size_t sa =
                    (static_cast<size_t>(batch[j].meta) << 1) & pair_mask;
                const size_t sb =
                    (static_cast<size_t>(batch[j + 1].meta) << 1) & pair_mask;
                _mm_prefetch(reinterpret_cast<const char *>(
                                 ents + ((static_cast<size_t>(
                                              batch[j + kServeD].meta)
                                          << 1) &
                                         pair_mask)),
                             _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(
                                 ents + ((static_cast<size_t>(
                                              batch[j + 1 + kServeD].meta)
                                          << 1) &
                                         pair_mask)),
                             _MM_HINT_T0);
                const __m128i ka = _mm_load_si128(
                    reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const char *>(ents + sa)));
                const __m128i kb = _mm_load_si128(
                    reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const char *>(ents + sb)));
                const __m128i va = _mm_load_si128(
                    reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const char *>(ents + sa) + 16));
                const __m128i vb = _mm_load_si128(
                    reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const char *>(ents + sb) + 16));
                const __m128i ta = _mm_xor_si128(ka, wantA);
                const __m128i tb = _mm_xor_si128(kb, wantB);
                if (_mm_testz_si128(ta, ta) && _mm_testz_si128(tb, tb)) {
                    const uint64_t vala =
                        static_cast<uint64_t>(_mm_cvtsi128_si64(va));
                    const uint64_t exta =
                        static_cast<uint64_t>(_mm_extract_epi64(va, 1));
                    const uint64_t valb =
                        static_cast<uint64_t>(_mm_cvtsi128_si64(vb));
                    const uint64_t extb =
                        static_cast<uint64_t>(_mm_extract_epi64(vb, 1));
                    const uint64_t aba = ((vala >> 3) & 0x1FFFFFFFull) |
                                         (vala & 0xFFFFFFFF00000000ull);
                    const uint64_t abb = ((valb >> 3) & 0x1FFFFFFFull) |
                                         (valb & 0xFFFFFFFF00000000ull);
                    uint32_t *pa = out + cur;
                    *reinterpret_cast<uint64_t *>(pa) = aba;
                    *reinterpret_cast<uint64_t *>(pa + 2) = exta;
                    cur += static_cast<size_t>(vala & 7);
                    uint32_t *pb = out + cur;
                    *reinterpret_cast<uint64_t *>(pb) = abb;
                    *reinterpret_cast<uint64_t *>(pb + 2) = extb;
                    cur += static_cast<size_t>(valb & 7);
                    j++;
                    continue;
                }
            }
        }
        const __m128i want = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&batch[j].k0));
        const uint64_t k1 = static_cast<uint64_t>(_mm_extract_epi64(want, 1));

        if (__builtin_expect((k1 >> 56) == 0, 0)) {
            break;
        }

        {
            const uint64_t hp = batch[j + kServeD].meta;
            _mm_prefetch(reinterpret_cast<const char *>(
                             ents + ((static_cast<size_t>(hp) << 1) &
                                     pair_mask)),
                         _MM_HINT_T0);
        }
        const size_t set =
            (static_cast<size_t>(batch[j].meta) << 1) & pair_mask;

        uint64_t val = 0, ext = 0;
        size_t ws = set;
        int hit = 0;
        for (;;) {
            const char *e0 = reinterpret_cast<const char *>(ents + ws);
            const __m128i k0v =
                _mm_load_si128(reinterpret_cast<const __m128i *>(e0));
            const __m128i v0 =
                _mm_load_si128(reinterpret_cast<const __m128i *>(e0 + 16));

            const __m128i t0 = _mm_xor_si128(k0v, want);
            if (_mm_testz_si128(t0, t0)) {
                val = static_cast<uint64_t>(_mm_cvtsi128_si64(v0));
                ext = static_cast<uint64_t>(_mm_extract_epi64(v0, 1));
                hit = 1;
                break;
            }
            const char *e1 = e0 + sizeof(Entry);
            const __m128i k1v =
                _mm_load_si128(reinterpret_cast<const __m128i *>(e1));
            const __m128i t1 = _mm_xor_si128(k1v, want);
            if (_mm_testz_si128(t1, t1)) {
                const __m128i v1 =
                    _mm_load_si128(reinterpret_cast<const __m128i *>(e1 + 16));
                val = static_cast<uint64_t>(_mm_cvtsi128_si64(v1));
                ext = static_cast<uint64_t>(_mm_extract_epi64(v1, 1));
                hit = 1;
                break;
            }

            if (_mm_testz_si128(k0v, k0v)) {
                break;
            }
            ws = (ws + 2) & mask;
        }
        if (!hit) {
            break;
        }

        const uint64_t ab = ((val >> 3) & 0x1FFFFFFFull) |
                            (val & 0xFFFFFFFF00000000ull);
        uint32_t *p = out + cur;
        *reinterpret_cast<uint64_t *>(p) = ab;
        *reinterpret_cast<uint64_t *>(p + 2) = ext;
        cur += static_cast<size_t>(val & 7);
    }
    *cur_inout = cur;
    return j;
}
