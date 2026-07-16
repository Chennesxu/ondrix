#include <stdint.h>
#include <stdio.h>

typedef int16_t (*kernel_t)(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                            int64_t, int64_t, int64_t);
typedef int16_t (*seeded_kernel_t)(int16_t, int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                   int16_t *, int16_t *, int64_t, int64_t, int64_t);

extern int16_t q15_auto_vector_saturate(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                        int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_auto_vector_wrap(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                    int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_seeded_vector_wrap(int16_t, int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                      int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_offset_vector_saturate(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                          int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_scalar_fallback_saturate(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                            int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_scalar_fallback_wrap(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                        int16_t *, int64_t, int64_t, int64_t);

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
  const __int128 updated = (__int128)accumulator + (__int128)lhs * (__int128)rhs;
  if (!saturate)
    return wrap_i40(updated);
  if (updated < minimum)
    return (int64_t)minimum;
  if (updated > maximum)
    return (int64_t)maximum;
  return (int64_t)updated;
}

static uint16_t reference_with_initial(const int16_t *lhs, int64_t lhs_offset, int64_t lhs_stride,
                                       const int16_t *rhs, int64_t rhs_offset, int64_t rhs_stride,
                                       int64_t length, int saturate, int64_t accumulator) {
  for (int64_t i = 0; i < length; ++i)
    accumulator = update_reference(accumulator, lhs[lhs_offset + i * lhs_stride],
                                   rhs[rhs_offset + i * rhs_stride], saturate);

  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  return (uint16_t)quotient;
}

static uint16_t reference(const int16_t *lhs, int64_t lhs_offset, int64_t lhs_stride,
                          const int16_t *rhs, int64_t rhs_offset, int64_t rhs_stride,
                          int64_t length, int saturate) {
  return reference_with_initial(lhs, lhs_offset, lhs_stride, rhs, rhs_offset, rhs_stride, length,
                                saturate, 0);
}

static uint16_t run(kernel_t kernel, int16_t *lhs, int64_t lhs_offset, int64_t lhs_stride,
                    int16_t *rhs, int64_t rhs_offset, int64_t rhs_stride, int64_t length) {
  return (uint16_t)kernel(lhs, lhs, lhs_offset, length, lhs_stride, rhs, rhs, rhs_offset, length,
                          rhs_stride);
}

static uint16_t run_seeded(seeded_kernel_t kernel, int16_t seed, int16_t *lhs, int16_t *rhs,
                           int64_t length) {
  return (uint16_t)kernel(seed, lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
}

static int check_case(const char *name, int16_t *lhs, int16_t *rhs, int64_t length, int saturate) {
  kernel_t vector_kernel = saturate ? q15_auto_vector_saturate : q15_auto_vector_wrap;
  kernel_t scalar_kernel = saturate ? q15_scalar_fallback_saturate : q15_scalar_fallback_wrap;
  const uint16_t expected = reference(lhs, 0, 1, rhs, 0, 1, length, saturate);
  const uint16_t vector_result = run(vector_kernel, lhs, 0, 1, rhs, 0, 1, length);
  const uint16_t scalar_result = run(scalar_kernel, lhs, 0, 1, rhs, 0, 1, length);
  if (vector_result == expected && scalar_result == expected)
    return 0;
  fprintf(stderr, "%s length %lld: expected 0x%04x, vector 0x%04x, scalar 0x%04x\n", name,
          (long long)length, expected, vector_result, scalar_result);
  return 1;
}

static int check_strided(int16_t *lhs, int16_t *rhs) {
  const int64_t length = 17;
  const uint16_t expected = reference(lhs, 1, 2, rhs, 2, 3, length, 1);
  const uint16_t actual = run(q15_scalar_fallback_saturate, lhs, 1, 2, rhs, 2, 3, length);
  if (actual == expected)
    return 0;
  fprintf(stderr, "strided fallback: expected 0x%04x, got 0x%04x\n", expected, actual);
  return 1;
}

static int check_seeded(int16_t *lhs, int16_t *rhs) {
  const int16_t seed = -12345;
  const int64_t length = 17;
  const int64_t initial = (int64_t)seed * (INT64_C(1) << 15);
  const uint16_t expected = reference_with_initial(lhs, 0, 1, rhs, 0, 1, length, 0, initial);
  const uint16_t actual = run_seeded(q15_seeded_vector_wrap, seed, lhs, rhs, length);
  if (actual == expected)
    return 0;
  fprintf(stderr, "seeded vector wrap: expected 0x%04x, got 0x%04x\n", expected, actual);
  return 1;
}

static int check_unit_stride_offset(int16_t *lhs, int16_t *rhs) {
  const int64_t length = 17;
  const uint16_t expected = reference(lhs, 1, 1, rhs, 2, 1, length, 1);
  const uint16_t actual = run(q15_offset_vector_saturate, lhs, 1, 1, rhs, 2, 1, length);
  if (actual == expected)
    return 0;
  fprintf(stderr, "unit-stride offset vector: expected 0x%04x, got 0x%04x\n", expected, actual);
  return 1;
}

int main(void) {
  int16_t lhs[520];
  int16_t rhs[520];
  uint32_t state = UINT32_C(0x243f6a88);
  for (unsigned i = 0; i < 520; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    lhs[i] = (int16_t)(state >> 16);
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    rhs[i] = (int16_t)(state >> 16);
  }

  static const int64_t lengths[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 511, 512, 513, 520};
  int failed = 0;
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    failed |= check_case("saturating", lhs, rhs, lengths[i], 1);
    failed |= check_case("wrapping", lhs, rhs, lengths[i], 0);
  }
  failed |= check_seeded(lhs, rhs);
  failed |= check_unit_stride_offset(lhs, rhs);

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
  failed |= check_case("order-sensitive saturating", lhs, rhs, 514, 1);
  failed |= check_case("overflow wrapping", lhs, rhs, 512, 0);

  for (unsigned i = 0; i < 520; ++i) {
    lhs[i] = (int16_t)(i * 97u + 11u);
    rhs[i] = (int16_t)(i * 53u - 7u);
  }
  failed |= check_strided(lhs, rhs);
  return failed;
}
