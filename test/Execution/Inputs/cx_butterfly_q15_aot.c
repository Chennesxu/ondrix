#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#ifdef OX_MULTI_RESULT
struct ButterflyResult {
  int32_t out0;
  int32_t out1;
};
extern void _mlir_ciface_q15_butterfly(struct ButterflyResult *result, int32_t a, int32_t b,
                                       int32_t twiddle);

static void execute_butterfly(int32_t a, int32_t b, int32_t twiddle, int32_t *out0, int32_t *out1) {
  struct ButterflyResult result;
  _mlir_ciface_q15_butterfly(&result, a, b, twiddle);
  *out0 = result.out0;
  *out1 = result.out1;
}
#else
extern int32_t cx_butterfly_q15_result(int32_t a, int32_t b, int32_t twiddle, int32_t result_index);

static void execute_butterfly(int32_t a, int32_t b, int32_t twiddle, int32_t *out0, int32_t *out1) {
  *out0 = cx_butterfly_q15_result(a, b, twiddle, 0);
  *out1 = cx_butterfly_q15_result(a, b, twiddle, 1);
}
#endif

struct Complex {
  int16_t real;
  int16_t imaginary;
};

static int16_t requantize(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }

  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
}

static int32_t toSigned32(uint32_t bits) {
  return bits <= (uint32_t)INT32_MAX ? (int32_t)bits
                                     : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
}

static int32_t pack(struct Complex value) {
  return toSigned32(((uint32_t)(uint16_t)value.imaginary << 16) | (uint32_t)(uint16_t)value.real);
}

static int16_t decodeSigned16(uint16_t bits) {
  int32_t value = bits;
  if ((bits & UINT16_C(0x8000)) != 0)
    value -= INT32_C(65536);
  return (int16_t)value;
}

static struct Complex unpack(int32_t value) {
  return (struct Complex){decodeSigned16((uint16_t)value),
                          decodeSigned16((uint16_t)((uint32_t)value >> 16))};
}

static void butterfly_reference(int32_t packed_a, int32_t packed_b, int32_t packed_twiddle,
                                int32_t *out0, int32_t *out1) {
  struct Complex a = unpack(packed_a);
  struct Complex b = unpack(packed_b);
  struct Complex w = unpack(packed_twiddle);
  __int128 product_real = (__int128)b.real * w.real - (__int128)b.imaginary * w.imaginary;
  __int128 product_imaginary = (__int128)b.real * w.imaginary + (__int128)b.imaginary * w.real;
  int16_t twiddled_real = requantize(product_real, 15);
  int16_t twiddled_imaginary = requantize(product_imaginary, 15);

  struct Complex first = {
      requantize((__int128)a.real + twiddled_real, 1),
      requantize((__int128)a.imaginary + twiddled_imaginary, 1),
  };
  struct Complex second = {
      requantize((__int128)a.real - twiddled_real, 1),
      requantize((__int128)a.imaginary - twiddled_imaginary, 1),
  };
  *out0 = pack(first);
  *out1 = pack(second);
}

struct Case {
  const char *name;
  struct Complex a;
  struct Complex b;
  struct Complex twiddle;
};

#if defined(TEST_VECTOR_CONSTANT_TWIDDLES) || defined(TEST_VECTOR_TABLE_TWIDDLES)
struct MemRefI32 {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};
#endif

#ifdef TEST_VECTOR_CONSTANT_TWIDDLES
typedef void (*VectorButterflyFn)(struct MemRefI32 *, struct MemRefI32 *, struct MemRefI32 *,
                                  struct MemRefI32 *);
extern void _mlir_ciface_bounded_vector_twiddles(struct MemRefI32 *, struct MemRefI32 *,
                                                 struct MemRefI32 *, struct MemRefI32 *);
extern void _mlir_ciface_unbounded_vector_twiddles(struct MemRefI32 *, struct MemRefI32 *,
                                                   struct MemRefI32 *, struct MemRefI32 *);

static int test_vector_constants(void) {
  static const int32_t twiddles[2][4] = {{-1518445950, -2147450881, 2147450880, INT32_MIN},
                                         {-1518445950, -2147450881, 2147450880, -2147450880}};
  static const VectorButterflyFn functions[] = {_mlir_ciface_bounded_vector_twiddles,
                                                _mlir_ciface_unbounded_vector_twiddles};
  static const int16_t edges[] = {INT16_MIN, INT16_MAX, -23170, -3, -1, 0, 1, 3, 16384};
  const unsigned edge_count = sizeof(edges) / sizeof(edges[0]);
  uint32_t random = 0x13579BDF;
  for (unsigned trial = 0; trial < 1024; ++trial) {
    int32_t a[4], b[4], out0[4], out1[4];
    for (unsigned lane = 0; lane < 4; ++lane) {
      if (trial < edge_count) {
        a[lane] = pack((struct Complex){edges[trial], edges[(trial + lane) % edge_count]});
        b[lane] = pack((struct Complex){edges[trial], edges[trial]});
      } else {
        random = random * 1664525U + 1013904223U;
        a[lane] = (int32_t)random;
        random = random * 1664525U + 1013904223U;
        b[lane] = (int32_t)random;
      }
    }
    struct MemRefI32 ma = {a, a, 0, 4, 1}, mb = {b, b, 0, 4, 1};
    struct MemRefI32 mo0 = {out0, out0, 0, 4, 1}, mo1 = {out1, out1, 0, 4, 1};
    for (unsigned variant = 0; variant < 2; ++variant) {
      functions[variant](&ma, &mb, &mo0, &mo1);
      for (unsigned lane = 0; lane < 4; ++lane) {
        int32_t expected0, expected1, scalar0, scalar1;
        butterfly_reference(a[lane], b[lane], twiddles[variant][lane], &expected0, &expected1);
        execute_butterfly(a[lane], b[lane], twiddles[variant][lane], &scalar0, &scalar1);
        if (out0[lane] != expected0 || out1[lane] != expected1 || scalar0 != expected0 ||
            scalar1 != expected1) {
          fprintf(stderr, "vector butterfly mismatch: trial %u, variant %u, lane %u\n", trial,
                  variant, lane);
          return 1;
        }
      }
    }
  }
  return 0;
}
#endif

#ifdef TEST_VECTOR_TABLE_TWIDDLES
typedef void (*TableButterflyFn)(struct MemRefI32 *, struct MemRefI32 *, struct MemRefI32 *,
                                 struct MemRefI32 *, int64_t);
extern void _mlir_ciface_safe_table_vector(struct MemRefI32 *, struct MemRefI32 *,
                                           struct MemRefI32 *, struct MemRefI32 *, int64_t);
extern void _mlir_ciface_unsafe_table_vector(struct MemRefI32 *, struct MemRefI32 *,
                                             struct MemRefI32 *, struct MemRefI32 *, int64_t);
extern void _mlir_ciface_masked_table_vector(struct MemRefI32 *, struct MemRefI32 *,
                                             struct MemRefI32 *, struct MemRefI32 *, int64_t);
extern void _mlir_ciface_padded_table_vector(struct MemRefI32 *, struct MemRefI32 *,
                                             struct MemRefI32 *, struct MemRefI32 *, int64_t);
extern void _mlir_ciface_safe_table_load_vector(struct MemRefI32 *, struct MemRefI32 *,
                                                struct MemRefI32 *, struct MemRefI32 *, int64_t);

static int test_vector_tables(void) {
  static const int32_t table[] = {32767,      -1518445950, INT32_MIN, -2147450881,
                                  2147450880, 32767,       0,         32767};
  static const TableButterflyFn functions[] = {
      _mlir_ciface_safe_table_vector, _mlir_ciface_unsafe_table_vector,
      _mlir_ciface_masked_table_vector, _mlir_ciface_padded_table_vector,
      _mlir_ciface_safe_table_load_vector};
  const int32_t unsafe_word = -2147450880;
  uint32_t random = 0x2468ACE1;
  for (unsigned trial = 0; trial < 1024; ++trial) {
    int32_t a[4], b[4], out0[4], out1[4];
    for (unsigned lane = 0; lane < 4; ++lane) {
      random = random * 1664525U + 1013904223U;
      a[lane] = trial < 2 ? 0 : (int32_t)random;
      random = random * 1664525U + 1013904223U;
      b[lane] = trial < 2 ? unsafe_word : (int32_t)random;
    }
    struct MemRefI32 ma = {a, a, 0, 4, 1}, mb = {b, b, 0, 4, 1};
    struct MemRefI32 mo0 = {out0, out0, 0, 4, 1}, mo1 = {out1, out1, 0, 4, 1};
    for (unsigned variant = 0; variant < sizeof(functions) / sizeof(functions[0]); ++variant) {
      int64_t index = variant == 3 ? 6 : variant == 4 ? 4 : (trial & 1) * 4;
      functions[variant](&ma, &mb, &mo0, &mo1, index);
      for (unsigned lane = 0; lane < 4; ++lane) {
        int64_t position = index + lane;
        int32_t twiddle = position < 8 ? table[position] : unsafe_word;
        if ((variant == 1 && position == 7) || (variant == 2 && lane == 3))
          twiddle = unsafe_word;
        int32_t expected0, expected1, scalar0, scalar1;
        butterfly_reference(a[lane], b[lane], twiddle, &expected0, &expected1);
        execute_butterfly(a[lane], b[lane], twiddle, &scalar0, &scalar1);
        if (out0[lane] != expected0 || out1[lane] != expected1 || scalar0 != expected0 ||
            scalar1 != expected1) {
          fprintf(stderr, "table butterfly mismatch: trial %u, variant %u, lane %u\n", trial,
                  variant, lane);
          return 1;
        }
      }
    }
  }
  return 0;
}
#endif

int main(void) {
  static const struct Case cases[] = {
      {"zero", {0, 0}, {0, 0}, {0, 0}},
      {"identity", {12000, -5000}, {7000, 9000}, {INT16_MAX, 0}},
      {"nontrivial", {-12000, 21000}, {18000, -23000}, {23170, -23170}},
      {"mixed-sign", {INT16_MIN, INT16_MAX}, {-12345, 23456}, {-30000, 11000}},
      {"min-cross-product", {0, 0}, {INT16_MIN, INT16_MIN}, {INT16_MIN, INT16_MIN}},
      {"positive-product-tie-even", {1, -1}, {1, 0}, {16384, 0}},
      {"positive-product-tie-odd", {3, -3}, {3, 0}, {16384, 0}},
      {"output-tie", {1, 3}, {0, 0}, {INT16_MAX, 0}},
  };

  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    int32_t a = pack(cases[i].a);
    int32_t b = pack(cases[i].b);
    int32_t twiddle = pack(cases[i].twiddle);
    int32_t expected0, expected1;
    butterfly_reference(a, b, twiddle, &expected0, &expected1);
    int32_t actual0, actual1;
    execute_butterfly(a, b, twiddle, &actual0, &actual1);
    if (actual0 != expected0 || actual1 != expected1) {
      fprintf(stderr,
              "%s: expected (%08" PRIx32 ", %08" PRIx32 "), got (%08" PRIx32 ", %08" PRIx32 ")\n",
              cases[i].name, (uint32_t)expected0, (uint32_t)expected1, (uint32_t)actual0,
              (uint32_t)actual1);
      return 1;
    }
  }
#ifdef TEST_VECTOR_CONSTANT_TWIDDLES
  if (test_vector_constants())
    return 1;
#endif
#ifdef TEST_VECTOR_TABLE_TWIDDLES
  return test_vector_tables();
#else
  return 0;
#endif
}
