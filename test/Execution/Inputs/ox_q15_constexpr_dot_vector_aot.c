#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>

extern int16_t q15_dot_constexpr(int16_t *, int16_t *, int64_t, int64_t, int64_t);

static int16_t dot_reference(const int16_t lhs[8]) {
  static const int16_t rhs[8] = {1, -2, 3, -4, 5, -6, 7, -8};
  const struct Policy policy = {
      .width = 16,
      .frac = 15,
      .accumulator_width = 40,
      .accumulator_frac = 30,
      .update_overflow = SATURATE,
      .state_rounding = NEAREST_EVEN,
      .state_overflow = SATURATE,
      .output_rounding = NEAREST_EVEN,
      .output_overflow = SATURATE,
  };
  int64_t accumulator = 0;
  for (int64_t i = 0; i < 8; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], &policy);
  return (int16_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int16_t inputs[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {-32768, 32767, 16384, -16384, 1, -1, 12345, -23456},
      {32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767},
      {-1, 1, -2, 2, -3, 3, -4, 4},
  };

  for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    int16_t actual = q15_dot_constexpr(inputs[i], inputs[i], 0, 8, 1);
    int16_t expected = dot_reference(inputs[i]);
    if (actual != expected) {
      fprintf(stderr, "constexpr Q15 dot case %u: got %d, expected %d\n", i, actual, expected);
      return 1;
    }
  }
  return 0;
}
