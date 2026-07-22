#include "fixed_point_reference.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t q31_dot(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                       int64_t, int64_t, int64_t);

static int32_t dot_reference(const int32_t *lhs, const int32_t *rhs, int64_t count) {
  const struct Policy policy = {
      .width = 32,
      .frac = 31,
      .accumulator_width = 64,
      .accumulator_frac = 62,
      .update_overflow = SATURATE,
      .state_rounding = NEAREST_EVEN,
      .state_overflow = SATURATE,
      .output_rounding = NEAREST_EVEN,
      .output_overflow = SATURATE,
  };
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], &policy);
  return (int32_t)export_reference(accumulator, policy.output_rounding, policy.output_overflow,
                                   &policy);
}

int main(void) {
  int32_t lhs[] = {INT32_MIN, INT32_MIN, INT32_MAX, INT32_C(1) << 30, -17};
  int32_t rhs[] = {INT32_MIN, INT32_MIN, INT32_MAX, -(INT32_C(1) << 30), 31};
  int64_t lengths[] = {0, 1, 2, 3, 5};

  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = lengths[i];
    int32_t actual = q31_dot(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
    int32_t expected = dot_reference(lhs, rhs, length);
    if (actual != expected) {
      fprintf(stderr, "q31 dot length %lld: got %d, expected %d\n", (long long)length, actual,
              expected);
      return 1;
    }
  }
  return 0;
}
