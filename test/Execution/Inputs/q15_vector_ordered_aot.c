#include <stdint.h>
#include <stdio.h>

extern int16_t q15_vector_reduce_saturate(int16_t *lhs_allocated, int16_t *lhs_aligned,
                                          int64_t lhs_offset, int64_t lhs_size, int64_t lhs_stride,
                                          int16_t *rhs_allocated, int16_t *rhs_aligned,
                                          int64_t rhs_offset, int64_t rhs_size, int64_t rhs_stride);
extern int16_t q15_vector_reduce_wrap(int16_t *lhs_allocated, int16_t *lhs_aligned,
                                      int64_t lhs_offset, int64_t lhs_size, int64_t lhs_stride,
                                      int16_t *rhs_allocated, int16_t *rhs_aligned,
                                      int64_t rhs_offset, int64_t rhs_size, int64_t rhs_stride);
extern int16_t q15_vector_i32_saturate(int16_t *lhs_allocated, int16_t *lhs_aligned,
                                       int64_t lhs_offset, int64_t lhs_size, int64_t lhs_stride,
                                       int16_t *rhs_allocated, int16_t *rhs_aligned,
                                       int64_t rhs_offset, int64_t rhs_size, int64_t rhs_stride);
extern int16_t q15_vector_i32_wrap(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                                   int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                                   int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                                   int64_t rhs_stride);

static int64_t wrap_i40(__int128 value) {
  const unsigned __int128 mask = ((unsigned __int128)1 << 40) - 1;
  const uint64_t bits = (uint64_t)((unsigned __int128)value & mask);
  __int128 signed_value = bits;
  if (bits & (UINT64_C(1) << 39))
    signed_value -= (__int128)1 << 40;
  return (int64_t)signed_value;
}

static int64_t update_reference(int64_t accumulator, int16_t lhs, int16_t rhs, int saturate) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 updated = (__int128)accumulator + (__int128)lhs * (__int128)rhs;
  if (!saturate)
    return wrap_i40(updated);
  if (updated < minimum)
    return (int64_t)minimum;
  if (updated > maximum)
    return (int64_t)maximum;
  return (int64_t)updated;
}

static uint16_t reference(const int16_t *lhs, const int16_t *rhs, int saturate) {
  int64_t accumulator = 0;
  for (unsigned i = 0; i < 520; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], saturate);

  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  return (uint16_t)quotient;
}

static int check(const char *name, int16_t *lhs, int16_t *rhs, int saturate) {
  int16_t actual_value = saturate
                             ? q15_vector_reduce_saturate(lhs, lhs, 0, 520, 1, rhs, rhs, 0, 520, 1)
                             : q15_vector_reduce_wrap(lhs, lhs, 0, 520, 1, rhs, rhs, 0, 520, 1);
  uint16_t actual = (uint16_t)actual_value;
  uint16_t expected = reference(lhs, rhs, saturate);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s: expected bits 0x%04x, got 0x%04x\n", name, expected, actual);
  return 1;
}

static int32_t update_i32_reference(int32_t accumulator, int16_t lhs, int16_t rhs, int saturate) {
  int64_t updated = (int64_t)accumulator + (int64_t)lhs * (int64_t)rhs;
  if (!saturate) {
    uint32_t bits = (uint32_t)updated;
    return bits <= INT32_MAX ? (int32_t)bits : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
  }
  if (updated < INT32_MIN)
    return INT32_MIN;
  if (updated > INT32_MAX)
    return INT32_MAX;
  return (int32_t)updated;
}

static int check_i32_ordered_vector(int16_t lhs[4], int16_t rhs[4], int saturate) {
  int32_t accumulator = 0;
  for (unsigned i = 0; i < 4; ++i)
    accumulator = update_i32_reference(accumulator, lhs[i], rhs[i], saturate);
  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  uint16_t expected = (uint16_t)quotient;
  uint16_t actual =
      (uint16_t)(saturate ? q15_vector_i32_saturate(lhs, lhs, 0, 4, 1, rhs, rhs, 0, 4, 1)
                          : q15_vector_i32_wrap(lhs, lhs, 0, 4, 1, rhs, rhs, 0, 4, 1));
  if (actual == expected)
    return 0;
  fprintf(stderr, "i32 %s ordered vector: expected bits 0x%04x, got 0x%04x\n",
          saturate ? "saturating" : "wrapping", expected, actual);
  return 1;
}

int main(void) {
  int16_t lhs[520];
  int16_t rhs[520];
  uint32_t state = UINT32_C(0x9e3779b9);

  for (unsigned i = 0; i < 520; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    lhs[i] = (int16_t)(state >> 16);
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    rhs[i] = (int16_t)(state >> 16);
  }
  int failed = check("random saturating vector reduction", lhs, rhs, 1);
  failed |= check("random wrapping vector reduction", lhs, rhs, 0);

  for (unsigned i = 0; i < 520; ++i) {
    lhs[i] = 0;
    rhs[i] = 0;
  }
  for (unsigned i = 0; i < 513; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  lhs[513] = INT16_MIN;
  rhs[513] = INT16_MAX;
  failed |= check("saturating order-sensitive vector reduction", lhs, rhs, 1);
  failed |= check("wrapping vector reduction", lhs, rhs, 0);

  int16_t i32_lhs[4] = {INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN};
  int16_t i32_rhs[4] = {INT16_MIN, INT16_MIN, INT16_MAX, INT16_MAX};
  failed |= check_i32_ordered_vector(i32_lhs, i32_rhs, 1);
  failed |= check_i32_ordered_vector(i32_lhs, i32_rhs, 0);
  return failed;
}
