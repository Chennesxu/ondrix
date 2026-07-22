#include "fixed_point_reference.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t q31_fir_constexpr(int32_t *, int32_t *, int64_t, int64_t, int64_t);

static int32_t fir_reference(const int32_t input[4]) {
  static const int32_t coefficients[4] = {INT32_C(1073741824), -INT32_C(536870912),
                                          -INT32_C(536870912), INT32_C(1073741824)};
  const struct Policy policy = {
      .width = 32,
      .frac = 31,
      .accumulator_width = 64,
      .accumulator_frac = 62,
      .update_overflow = WRAP,
      .state_rounding = NEAREST_EVEN,
      .state_overflow = SATURATE,
      .output_rounding = NEAREST_EVEN,
      .output_overflow = SATURATE,
  };
  int64_t accumulator = 0;
  for (int64_t i = 0; i < 4; ++i)
    accumulator = update_reference(accumulator, input[i], coefficients[i], &policy);
  return (int32_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int32_t inputs[][4] = {
      {0, 0, 0, 0},
      {INT32_MIN, INT32_MAX, INT32_C(1) << 30, -(INT32_C(1) << 30)},
      {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX},
      {-123456789, 987654321, -1, 1},
  };

  for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    int32_t actual = q31_fir_constexpr(inputs[i], inputs[i], 0, 4, 1);
    int32_t expected = fir_reference(inputs[i]);
    if (actual != expected) {
      fprintf(stderr, "constexpr Q31 FIR case %u: got %d, expected %d\n", i, actual, expected);
      return 1;
    }
  }
  return 0;
}
