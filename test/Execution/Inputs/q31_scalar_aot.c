#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t q31_full_nearest_even(int32_t lhs, int32_t rhs);
extern int32_t q31_repeat_full_saturate(int32_t lhs, int32_t rhs, int64_t count);
extern int32_t q31_repeat_full_wrap(int32_t lhs, int32_t rhs, int64_t count);
extern int32_t q31_high_raw_q30(int32_t lhs, int32_t rhs);
extern int32_t q31_high_raw_sub_q30(int32_t lhs, int32_t rhs);
extern int32_t q31_reduce_full(int32_t *lhs_allocated, int32_t *lhs_aligned, int64_t lhs_offset,
                               int64_t lhs_size, int64_t lhs_stride, int32_t *rhs_allocated,
                               int32_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                               int64_t rhs_stride);
extern int32_t q31_reduce_high_raw_q30(int32_t *lhs_allocated, int32_t *lhs_aligned,
                                       int64_t lhs_offset, int64_t lhs_size, int64_t lhs_stride,
                                       int32_t *rhs_allocated, int32_t *rhs_aligned,
                                       int64_t rhs_offset, int64_t rhs_size, int64_t rhs_stride);
extern int64_t q31_repeat_full_raw_saturate(int32_t lhs, int32_t rhs, int64_t count);
extern int64_t q31_repeat_high_raw_saturate(int32_t lhs, int32_t rhs, int64_t count);

static __int128 signed_i64(uint64_t bits) {
  __int128 value = bits;
  if (bits & (UINT64_C(1) << 63))
    value -= (__int128)1 << 64;
  return value;
}

static __int128 floor_divide_by_power_of_two(__int128 value, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

static int32_t saturate_i32(__int128 value) {
  if (value < INT32_MIN)
    return INT32_MIN;
  if (value > INT32_MAX)
    return INT32_MAX;
  return (int32_t)value;
}

static uint64_t update_full_reference(uint64_t accumulator_bits, int32_t lhs, int32_t rhs,
                                      int saturate) {
  const __int128 minimum = -((__int128)1 << 63);
  const __int128 maximum = ((__int128)1 << 63) - 1;
  __int128 updated = signed_i64(accumulator_bits) + (__int128)lhs * (__int128)rhs;
  if (saturate) {
    if (updated < minimum)
      updated = minimum;
    if (updated > maximum)
      updated = maximum;
  }
  return (uint64_t)updated;
}

static int32_t export_q31_reference(uint64_t accumulator_bits, int nearest_even) {
  __int128 accumulator = signed_i64(accumulator_bits);
  __int128 quotient = floor_divide_by_power_of_two(accumulator, 31);
  if (nearest_even) {
    __int128 divisor = (__int128)1 << 31;
    __int128 remainder = accumulator - quotient * divisor;
    __int128 half = divisor / 2;
    if (remainder > half || (remainder == half && quotient % 2 != 0))
      ++quotient;
  }
  return saturate_i32(quotient);
}

static int32_t full_mul_reference(int32_t lhs, int32_t rhs) {
  uint64_t accumulator = update_full_reference(0, lhs, rhs, 1);
  return export_q31_reference(accumulator, 1);
}

static int32_t repeat_full_reference(int32_t lhs, int32_t rhs, int64_t count, int saturate) {
  uint64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_full_reference(accumulator, lhs, rhs, saturate);
  return export_q31_reference(accumulator, 0);
}

static int32_t raw_high_reference(int32_t lhs, int32_t rhs) {
  __int128 product = (__int128)lhs * (__int128)rhs;
  return (int32_t)floor_divide_by_power_of_two(product, 32);
}

static int32_t reduce_full_reference(const int32_t *lhs, const int32_t *rhs, int64_t count) {
  uint64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_full_reference(accumulator, lhs[i], rhs[i], 1);
  return export_q31_reference(accumulator, 0);
}

static int32_t reduce_high_raw_reference(const int32_t *lhs, const int32_t *rhs, int64_t count) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 accumulator = 0;
  for (int64_t i = 0; i < count; ++i) {
    accumulator += raw_high_reference(lhs[i], rhs[i]);
    if (accumulator < minimum)
      accumulator = minimum;
    if (accumulator > maximum)
      accumulator = maximum;
  }
  return saturate_i32(accumulator);
}

static int64_t repeat_full_raw_reference(int32_t lhs, int32_t rhs, int64_t count) {
  uint64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_full_reference(accumulator, lhs, rhs, 1);
  return (int64_t)signed_i64(accumulator);
}

static int64_t repeat_high_raw_reference(int32_t lhs, int32_t rhs, int64_t count) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 accumulator = 0;
  for (int64_t i = 0; i < count; ++i) {
    accumulator += raw_high_reference(lhs, rhs);
    if (accumulator < minimum)
      accumulator = minimum;
    if (accumulator > maximum)
      accumulator = maximum;
  }
  return (int64_t)accumulator;
}

static int check_value(const char *name, int32_t expected, int32_t actual) {
  if (expected == actual)
    return 0;
  fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
  return 1;
}

static int check_wide_value(const char *name, int64_t expected, int64_t actual) {
  if (expected == actual)
    return 0;
  fprintf(stderr, "%s: expected %lld, got %lld\n", name, (long long)expected, (long long)actual);
  return 1;
}

static int check_full_products(void) {
  static const int32_t cases[][2] = {
      {0, 0},
      {INT32_MAX, INT32_MAX},
      {INT32_MIN, INT32_MIN},
      {INT32_MIN, INT32_MAX},
      {1, INT32_C(1) << 30},
      {3, INT32_C(1) << 30},
      {-1, INT32_C(1) << 30},
      {-3, INT32_C(1) << 30},
  };
  int failed = 0;
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    int32_t lhs = cases[i][0];
    int32_t rhs = cases[i][1];
    failed |= check_value("q31 full product", full_mul_reference(lhs, rhs),
                          q31_full_nearest_even(lhs, rhs));
  }
  return failed;
}

static int check_repeated_full_products(void) {
  static const int64_t counts[] = {0, 1, 2, 3};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
    int64_t count = counts[i];
    failed |=
        check_value("q31 saturating updates", repeat_full_reference(INT32_MIN, INT32_MIN, count, 1),
                    q31_repeat_full_saturate(INT32_MIN, INT32_MIN, count));
    failed |=
        check_value("q31 wrapping updates", repeat_full_reference(INT32_MIN, INT32_MIN, count, 0),
                    q31_repeat_full_wrap(INT32_MIN, INT32_MIN, count));
  }
  return failed;
}

static int check_raw_high_products(void) {
  static const int32_t cases[][2] = {
      {INT32_MAX, INT32_MAX},
      {INT32_MIN, INT32_MIN},
      {INT32_MIN, INT32_MAX},
      {INT32_MIN, 1},
      {-1, 1},
  };
  int failed = 0;
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    int32_t lhs = cases[i][0];
    int32_t rhs = cases[i][1];
    int32_t expected = raw_high_reference(lhs, rhs);
    failed |= check_value("q31 raw high", expected, q31_high_raw_q30(lhs, rhs));
    failed |= check_value("q31 raw high subtract", -expected, q31_high_raw_sub_q30(lhs, rhs));
  }
  return failed;
}

static int check_random_products(void) {
  uint32_t state = UINT32_C(0x9e3779b9);
  int failed = 0;
  for (unsigned i = 0; i < 64; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int32_t lhs = (int32_t)state;
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int32_t rhs = (int32_t)state;
    failed |= check_value("random q31 full", full_mul_reference(lhs, rhs),
                          q31_full_nearest_even(lhs, rhs));
    failed |= check_value("random q31 raw high", raw_high_reference(lhs, rhs),
                          q31_high_raw_q30(lhs, rhs));
  }
  return failed;
}

static int check_reductions(void) {
  static const int64_t lengths[] = {0, 1, 3, 8};
  int32_t lhs[] = {INT32_MIN, INT32_MIN, INT32_MIN, INT32_MAX, -123456789, 987654321, 1, -1};
  int32_t rhs[] = {INT32_MIN, INT32_MIN,  INT32_MAX, INT32_MAX,
                   333333333, -777777777, INT32_MIN, INT32_MAX};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t count = lengths[i];
    int32_t full = q31_reduce_full(lhs, lhs, 0, count, 1, rhs, rhs, 0, count, 1);
    int32_t high = q31_reduce_high_raw_q30(lhs, lhs, 0, count, 1, rhs, rhs, 0, count, 1);
    failed |= check_value("q31 full reduction", reduce_full_reference(lhs, rhs, count), full);
    failed |=
        check_value("q31 raw-high reduction", reduce_high_raw_reference(lhs, rhs, count), high);
  }
  return failed;
}

static int check_raw_accumulator_endpoints(void) {
  int failed = 0;
  static const int64_t full_counts[] = {0, 1, 2, 3};
  for (unsigned i = 0; i < sizeof(full_counts) / sizeof(full_counts[0]); ++i) {
    int64_t count = full_counts[i];
    failed |= check_wide_value("q31 full raw accumulator",
                               repeat_full_raw_reference(INT32_MIN, INT32_MIN, count),
                               q31_repeat_full_raw_saturate(INT32_MIN, INT32_MIN, count));
  }

  static const int64_t high_counts[] = {0, 1, 511, 512, 513};
  for (unsigned i = 0; i < sizeof(high_counts) / sizeof(high_counts[0]); ++i) {
    int64_t count = high_counts[i];
    failed |= check_wide_value("q31 raw-high accumulator",
                               repeat_high_raw_reference(INT32_MIN, INT32_MIN, count),
                               q31_repeat_high_raw_saturate(INT32_MIN, INT32_MIN, count));
  }
  return failed;
}

int main(void) {
  int failed = 0;
  failed |= check_full_products();
  failed |= check_repeated_full_products();
  failed |= check_raw_high_products();
  failed |= check_random_products();
  failed |= check_reductions();
  failed |= check_raw_accumulator_endpoints();
  return failed;
}
