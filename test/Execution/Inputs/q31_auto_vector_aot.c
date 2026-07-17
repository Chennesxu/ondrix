#include <limits.h>
#include <stdint.h>
#include <stdio.h>

typedef int64_t (*raw_kernel_t)(int32_t, int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                int32_t *, int64_t, int64_t, int64_t);

extern int64_t q31_vector_full_raw_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t);
extern int64_t q31_vector_full_raw_wrap(int32_t, int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                        int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern int64_t q31_vector_high_raw_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t);
extern int64_t q31_vector_high_raw_wrap(int32_t, int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                        int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern int64_t q31_vector_high_raw_offset_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                                   int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                                   int64_t);
extern int64_t q31_scalar_full_raw_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t);
extern int64_t q31_scalar_full_raw_wrap(int32_t, int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                        int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern int64_t q31_scalar_high_raw_saturate(int32_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t, int32_t *, int32_t *, int64_t, int64_t,
                                            int64_t);
extern int64_t q31_scalar_high_raw_wrap(int32_t, int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                        int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern int32_t q31_vector_full_export(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                      int32_t *, int64_t, int64_t, int64_t);
extern int32_t q31_vector_high_raw_export_q30(int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                              int32_t *, int32_t *, int64_t, int64_t, int64_t);

static __int128 signed_bits(uint64_t bits, unsigned width) {
  if (width < 64)
    bits &= (UINT64_C(1) << width) - 1;
  __int128 value = bits;
  if (bits & (UINT64_C(1) << (width - 1)))
    value -= (__int128)1 << width;
  return value;
}

static uint64_t truncate_bits(__int128 value, unsigned width) {
  uint64_t bits = (uint64_t)value;
  if (width < 64)
    bits &= (UINT64_C(1) << width) - 1;
  return bits;
}

static __int128 floor_divide_by_power_of_two(__int128 value, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

static __int128 high_raw_product(int32_t lhs, int32_t rhs) {
  return floor_divide_by_power_of_two((__int128)lhs * (__int128)rhs, 32);
}

static uint64_t update_reference(uint64_t accumulator_bits, unsigned width, __int128 term,
                                 int saturate) {
  __int128 updated = signed_bits(accumulator_bits, width) + term;
  if (saturate) {
    const __int128 minimum = -((__int128)1 << (width - 1));
    const __int128 maximum = ((__int128)1 << (width - 1)) - 1;
    if (updated < minimum)
      updated = minimum;
    if (updated > maximum)
      updated = maximum;
  }
  return truncate_bits(updated, width);
}

static int64_t reduce_reference(int32_t seed, const int32_t *lhs, int64_t lhs_stride,
                                const int32_t *rhs, int64_t rhs_stride, int64_t count, int high_raw,
                                int saturate) {
  const unsigned width = high_raw ? 40 : 64;
  __int128 seed_value = high_raw ? (__int128)seed : (__int128)seed * ((__int128)1 << 31);
  uint64_t accumulator = truncate_bits(seed_value, width);
  for (int64_t i = 0; i < count; ++i) {
    __int128 term = high_raw ? high_raw_product(lhs[i * lhs_stride], rhs[i * rhs_stride])
                             : (__int128)lhs[i * lhs_stride] * (__int128)rhs[i * rhs_stride];
    accumulator = update_reference(accumulator, width, term, saturate);
  }
  return (int64_t)signed_bits(accumulator, width);
}

static int32_t export_q31_nearest_even(int64_t accumulator) {
  __int128 value = accumulator;
  __int128 quotient = floor_divide_by_power_of_two(value, 31);
  const __int128 divisor = (__int128)1 << 31;
  const __int128 remainder = value - quotient * divisor;
  const __int128 half = divisor / 2;
  if (remainder > half || (remainder == half && (quotient & 1) != 0))
    ++quotient;
  if (quotient < INT32_MIN)
    return INT32_MIN;
  if (quotient > INT32_MAX)
    return INT32_MAX;
  return (int32_t)quotient;
}

static int32_t saturate_i32(int64_t value) {
  if (value < INT32_MIN)
    return INT32_MIN;
  if (value > INT32_MAX)
    return INT32_MAX;
  return (int32_t)value;
}

static int64_t run(raw_kernel_t kernel, int32_t seed, int32_t *lhs, int64_t lhs_offset,
                   int64_t lhs_stride, int32_t *rhs, int64_t rhs_offset, int64_t rhs_stride,
                   int64_t count) {
  return kernel(seed, lhs, lhs, lhs_offset, count, lhs_stride, rhs, rhs, rhs_offset, count,
                rhs_stride);
}

static int check_raw(const char *name, int64_t expected, int64_t vector, int64_t scalar) {
  if (expected == vector && expected == scalar)
    return 0;
  fprintf(stderr, "%s: expected %lld, vector %lld, scalar %lld\n", name, (long long)expected,
          (long long)vector, (long long)scalar);
  return 1;
}

static int check_domain(const char *name, raw_kernel_t vector, raw_kernel_t scalar, int32_t seed,
                        int32_t *lhs, int32_t *rhs, int64_t count, int high_raw, int saturate) {
  int64_t expected = reduce_reference(seed, lhs, 1, rhs, 1, count, high_raw, saturate);
  return check_raw(name, expected, run(vector, seed, lhs, 0, 1, rhs, 0, 1, count),
                   run(scalar, seed, lhs, 0, 1, rhs, 0, 1, count));
}

static int check_lengths(void) {
  static const int64_t lengths[] = {0, 1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 511, 512, 513};
  int32_t lhs[520];
  int32_t rhs[520];
  for (unsigned i = 0; i < 520; ++i) {
    lhs[i] = INT32_MIN;
    rhs[i] = INT32_MIN;
  }

  int failed = 0;
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t count = lengths[i];
    failed |=
        check_domain("q31 full saturate", q31_vector_full_raw_saturate,
                     q31_scalar_full_raw_saturate, INT32_C(0x12345678), lhs, rhs, count, 0, 1);
    failed |= check_domain("q31 full wrap", q31_vector_full_raw_wrap, q31_scalar_full_raw_wrap,
                           -INT32_C(0x1234567), lhs, rhs, count, 0, 0);
    failed |=
        check_domain("q31 high raw saturate", q31_vector_high_raw_saturate,
                     q31_scalar_high_raw_saturate, INT32_C(0x12345678), lhs, rhs, count, 1, 1);
    failed |= check_domain("q31 high raw wrap", q31_vector_high_raw_wrap, q31_scalar_high_raw_wrap,
                           -INT32_C(0x1234567), lhs, rhs, count, 1, 0);
  }
  return failed;
}

static int check_order_and_offset(void) {
  // The opposing term is lane 2 of the same vector chunk that overflows.
  int32_t full_lhs[4] = {INT32_MIN, INT32_MIN, INT32_MIN, 0};
  int32_t full_rhs[4] = {INT32_MIN, INT32_MIN, INT32_MAX, 0};
  int failed = check_domain("q31 full ordered saturation", q31_vector_full_raw_saturate,
                            q31_scalar_full_raw_saturate, 0, full_lhs, full_rhs, 4, 0, 1);

  int32_t high_lhs[516];
  int32_t high_rhs[516];
  for (unsigned i = 0; i < 516; ++i) {
    high_lhs[i] = INT32_MIN;
    high_rhs[i] = INT32_MIN;
  }
  // The final chunk starts saturated, then applies positive, negative, zero,
  // zero terms in lane order.
  high_rhs[513] = INT32_MAX;
  high_lhs[514] = high_rhs[514] = 0;
  high_lhs[515] = high_rhs[515] = 0;
  failed |= check_domain("q31 high raw ordered saturation", q31_vector_high_raw_saturate,
                         q31_scalar_high_raw_saturate, 0, high_lhs, high_rhs, 516, 1, 1);

  int32_t lhs[11];
  int32_t rhs[11];
  for (unsigned i = 0; i < 11; ++i) {
    lhs[i] = (int32_t)(UINT32_C(0x1020304) * (i + 1));
    rhs[i] = (i & 1) ? -(int32_t)(UINT32_C(0x7654321) * (i + 1))
                     : (int32_t)(UINT32_C(0x7654321) * (i + 1));
  }
  int64_t expected = reduce_reference(17, lhs + 1, 1, rhs + 2, 1, 8, 1, 1);
  int64_t vector = run(q31_vector_high_raw_offset_saturate, 17, lhs, 1, 1, rhs, 2, 1, 8);
  int64_t scalar = run(q31_scalar_high_raw_saturate, 17, lhs, 1, 1, rhs, 2, 1, 8);
  failed |= check_raw("q31 vector offset", expected, vector, scalar);

  int32_t strided_lhs[18];
  int32_t strided_rhs[18];
  for (unsigned i = 0; i < 18; ++i) {
    strided_lhs[i] = (int32_t)(UINT32_C(0x1111111) * (i + 1));
    strided_rhs[i] = (i & 1) ? INT32_MIN : INT32_MAX;
  }
  expected = reduce_reference(-23, strided_lhs + 1, 2, strided_rhs + 2, 2, 8, 1, 1);
  scalar = run(q31_scalar_high_raw_saturate, -23, strided_lhs, 1, 2, strided_rhs, 2, 2, 8);
  failed |= check_raw("q31 scalar stride fallback", expected, scalar, scalar);
  return failed;
}

static int check_random(void) {
  int32_t lhs[37];
  int32_t rhs[37];
  uint32_t state = UINT32_C(0x9e3779b9);
  for (unsigned i = 0; i < 37; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    lhs[i] = (int32_t)state;
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    rhs[i] = (int32_t)state;
  }
  int failed = 0;
  failed |= check_domain("random q31 full saturate", q31_vector_full_raw_saturate,
                         q31_scalar_full_raw_saturate, 12345, lhs, rhs, 37, 0, 1);
  failed |= check_domain("random q31 full wrap", q31_vector_full_raw_wrap, q31_scalar_full_raw_wrap,
                         -23456, lhs, rhs, 37, 0, 0);
  failed |= check_domain("random q31 high saturate", q31_vector_high_raw_saturate,
                         q31_scalar_high_raw_saturate, 34567, lhs, rhs, 37, 1, 1);
  failed |= check_domain("random q31 high wrap", q31_vector_high_raw_wrap, q31_scalar_high_raw_wrap,
                         -45678, lhs, rhs, 37, 1, 0);
  return failed;
}

static int check_exports(void) {
  int32_t lhs[9] = {INT32_MIN, INT32_MAX, 1, -1, 123456789, -987654321, INT32_MIN, INT32_MAX, 17};
  int32_t rhs[9] = {INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, -333333333,
                    777777777, INT32_MAX, INT32_MIN, -19};
  int64_t full_raw = reduce_reference(0, lhs, 1, rhs, 1, 9, 0, 1);
  int32_t full_expected = export_q31_nearest_even(full_raw);
  int32_t full_actual = q31_vector_full_export(lhs, lhs, 0, 9, 1, rhs, rhs, 0, 9, 1);

  int64_t high_raw = reduce_reference(0, lhs + 1, 1, rhs + 1, 1, 8, 1, 1);
  int32_t high_expected = saturate_i32(high_raw);
  int32_t high_actual = q31_vector_high_raw_export_q30(lhs, lhs, 1, 8, 1, rhs, rhs, 1, 8, 1);
  if (full_expected == full_actual && high_expected == high_actual)
    return 0;
  fprintf(stderr, "q31 exports: full expected %d got %d; high expected %d got %d\n", full_expected,
          full_actual, high_expected, high_actual);
  return 1;
}

int main(void) {
  int failed = 0;
  failed |= check_lengths();
  failed |= check_order_and_offset();
  failed |= check_random();
  failed |= check_exports();
  return failed;
}
