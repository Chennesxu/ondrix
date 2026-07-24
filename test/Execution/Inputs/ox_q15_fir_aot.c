#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>

#ifdef OX_AUTO_PROFILE
extern int16_t q15_fir_auto(int16_t *input_allocated, int16_t *input_aligned, int64_t input_offset,
                            int64_t input_size, int64_t input_stride, int16_t *coeffs_allocated,
                            int16_t *coeffs_aligned, int64_t coeffs_offset, int64_t coeffs_size,
                            int64_t coeffs_stride);
#define Q15_FIR q15_fir_auto
#define ACCUMULATOR_WIDTH 34
#define UPDATE_OVERFLOW WRAP
#else
extern int16_t q15_fir(int16_t *input_allocated, int16_t *input_aligned, int64_t input_offset,
                       int64_t input_size, int64_t input_stride, int16_t *coeffs_allocated,
                       int16_t *coeffs_aligned, int64_t coeffs_offset, int64_t coeffs_size,
                       int64_t coeffs_stride);
#define Q15_FIR q15_fir
#define ACCUMULATOR_WIDTH 40
#define UPDATE_OVERFLOW SATURATE
#endif

static int16_t fir_reference(const int16_t *input, const int16_t *coefficients, int64_t count) {
  const struct Policy policy = {
      .width = 16,
      .frac = 15,
      .accumulator_width = ACCUMULATOR_WIDTH,
      .accumulator_frac = 30,
      .update_overflow = UPDATE_OVERFLOW,
      .state_rounding = NEAREST_EVEN,
      .state_overflow = SATURATE,
      .output_rounding = NEAREST_EVEN,
      .output_overflow = SATURATE,
  };
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, input[i], coefficients[i], &policy);
  return (int16_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int16_t input[] = {-32768, 32767, 16384, -16384, 1, -1, 12345, -23456, 32767};
  int16_t coefficients[] = {-32768, 32767, 16384, 16384, 16384, 16384, -22222, 11111, -32768};
#ifdef OX_AUTO_PROFILE
  int64_t lengths[] = {4};
#else
  int64_t lengths[] = {0, 1, 2, 3, 8, 9};
#endif

  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = lengths[i];
    int16_t actual = Q15_FIR(input, input, 0, length, 1, coefficients, coefficients, 0, length, 1);
    int16_t expected = fir_reference(input, coefficients, length);
    if (actual != expected) {
      fprintf(stderr, "q15 FIR length %lld: got %d, expected %d\n", (long long)length, actual,
              expected);
      return 1;
    }
  }
  return 0;
}
