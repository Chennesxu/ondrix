#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>

extern int16_t q15_dot(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                       int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                       int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                       int64_t rhs_stride);

static int16_t dot_reference(const int16_t *lhs, const int16_t *rhs, int64_t count) {
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
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], &policy);
  return (int16_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int16_t lhs[] = {-32768, 32767, 16384, -16384, 1, -1, 12345, -23456, 32767};
  int16_t rhs[] = {-32768, 32767, 16384, 16384, 16384, 16384, -22222, 11111, -32768};
  int64_t lengths[] = {0, 1, 2, 3, 8, 9};

  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = lengths[i];
    int16_t actual = q15_dot(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
    int16_t expected = dot_reference(lhs, rhs, length);
    if (actual != expected) {
      fprintf(stderr, "q15 dot length %lld: got %d, expected %d\n", (long long)length, actual,
              expected);
      return 1;
    }
  }
  return 0;
}
