#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>

extern int16_t q15_fir_constexpr(int16_t *allocated, int16_t *aligned, int64_t offset, int64_t size,
                                 int64_t stride);

static int16_t fir_reference(const int16_t input[5]) {
  static const int16_t coefficients[5] = {16384, -8192, 4096, -8192, 16384};
  const struct Policy policy = {
      .width = 16,
      .frac = 15,
      .accumulator_width = 40,
      .accumulator_frac = 30,
      .update_overflow = WRAP,
      .state_rounding = NEAREST_EVEN,
      .state_overflow = SATURATE,
      .output_rounding = NEAREST_EVEN,
      .output_overflow = SATURATE,
  };
  int64_t accumulator = 0;
  for (int64_t i = 0; i < 5; ++i)
    accumulator = update_reference(accumulator, input[i], coefficients[i], &policy);
  return (int16_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int16_t inputs[][5] = {
      {0, 0, 0, 0, 0},
      {-32768, 32767, 16384, -16384, 1},
      {32767, 32767, 32767, 32767, 32767},
      {-12345, 23456, -1, 1, 30000},
  };

  for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    int16_t actual = q15_fir_constexpr(inputs[i], inputs[i], 0, 5, 1);
    int16_t expected = fir_reference(inputs[i]);
    if (actual != expected) {
      fprintf(stderr, "constexpr Q15 FIR case %u: got %d, expected %d\n", i, actual, expected);
      return 1;
    }
  }
  return 0;
}
