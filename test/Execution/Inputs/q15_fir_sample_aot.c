#include <stdint.h>
#include <stdio.h>

typedef int16_t (*fir_kernel)(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                              int64_t, int64_t, int64_t);

extern int16_t fir_q15_saturate(int16_t *input_allocated, int16_t *input_aligned,
                                int64_t input_offset, int64_t input_size, int64_t input_stride,
                                int16_t *coeffs_allocated, int16_t *coeffs_aligned,
                                int64_t coeffs_offset, int64_t coeffs_size, int64_t coeffs_stride);
extern int16_t fir_q15_wrap(int16_t *input_allocated, int16_t *input_aligned, int64_t input_offset,
                            int64_t input_size, int64_t input_stride, int16_t *coeffs_allocated,
                            int16_t *coeffs_aligned, int64_t coeffs_offset, int64_t coeffs_size,
                            int64_t coeffs_stride);
extern int16_t dot_q15_saturate(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                                int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                                int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                                int64_t rhs_stride);

static int64_t update_reference(int64_t accumulator, int16_t lhs, int16_t rhs, int saturate) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  __int128 updated = (__int128)accumulator + (__int128)lhs * (__int128)rhs;
  if (saturate) {
    if (updated < minimum)
      return (int64_t)minimum;
    if (updated > maximum)
      return (int64_t)maximum;
    return (int64_t)updated;
  }

  const uint64_t mask = (UINT64_C(1) << 40) - 1;
  uint64_t bits = (uint64_t)updated & mask;
  return (bits & (UINT64_C(1) << 39)) ? (int64_t)(bits - (UINT64_C(1) << 40)) : (int64_t)bits;
}

static uint16_t export_floor_wrap_reference(int64_t accumulator) {
  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  return (uint16_t)quotient;
}

static uint16_t fir_reference(const int16_t *input, const int16_t *coeffs, int64_t count,
                              int saturate) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, input[i], coeffs[i], saturate);
  return export_floor_wrap_reference(accumulator);
}

static int check(const char *name, fir_kernel kernel, int16_t *input, int16_t *coeffs,
                 int64_t count, int saturate) {
  uint16_t actual = (uint16_t)kernel(input, input, 0, count, 1, coeffs, coeffs, 0, count, 1);
  uint16_t expected = fir_reference(input, coeffs, count, saturate);
  if (actual == expected)
    return 0;
  fprintf(stderr, "%s(%lld): expected bits 0x%04x, got 0x%04x\n", name, (long long)count, expected,
          actual);
  return 1;
}

int main(void) {
  static const int64_t lengths[] = {0, 1, 2, 8, 511, 512, 513};
  int16_t input[513];
  int16_t coeffs[513];
  int failed = 0;

  for (unsigned i = 0; i < 513; ++i) {
    input[i] = INT16_MIN;
    coeffs[i] = INT16_MIN;
  }
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    failed |= check("fir saturate", fir_q15_saturate, input, coeffs, lengths[i], 1);
    failed |= check("fir wrap", fir_q15_wrap, input, coeffs, lengths[i], 0);
    failed |= check("dot saturate", dot_q15_saturate, input, coeffs, lengths[i], 1);
  }

  uint32_t state = UINT32_C(0x243f6a88);
  for (unsigned sample = 0; sample < 32; ++sample) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    int64_t count = state % 514;
    for (int64_t i = 0; i < count; ++i) {
      state = state * UINT32_C(1664525) + UINT32_C(1013904223);
      input[i] = (int16_t)(state >> 16);
      state = state * UINT32_C(1664525) + UINT32_C(1013904223);
      coeffs[i] = (int16_t)(state >> 16);
    }
    failed |= check("random fir saturate", fir_q15_saturate, input, coeffs, count, 1);
    failed |= check("random fir wrap", fir_q15_wrap, input, coeffs, count, 0);
    failed |= check("random dot saturate", dot_q15_saturate, input, coeffs, count, 1);
  }
  return failed;
}
