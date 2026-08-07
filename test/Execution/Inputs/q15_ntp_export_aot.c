#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>

extern int32_t ondsp_repeat_mac_ntp(int16_t lhs, int16_t rhs, int64_t count);
extern int16_t ondsp_repeat_mac_ntp_i16(int16_t lhs, int16_t rhs, int64_t count);

static const struct Policy policy32 = {
    .width = 32,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_TIES_POSITIVE,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_TIES_POSITIVE,
    .output_overflow = SATURATE,
};

static struct Policy policy16;

static int64_t accumulate(int16_t lhs, int16_t rhs, int64_t count, const struct Policy *policy) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs, rhs, policy);
  return accumulator;
}

struct Case {
  const char *name;
  int16_t lhs;
  int16_t rhs;
  int64_t count;
};

static const struct Case cases[] = {
    {"saturated positive rail", INT16_MIN, INT16_MIN, 512},
    {"held at the rail", INT16_MIN, INT16_MIN, 513},
    {"saturated negative rail", INT16_MIN, INT16_MAX, 512},
    {"negative half tie", 1, -16384, 3},
    {"positive half tie, even quotient", 1, 16384, 5},
    {"positive half tie, odd quotient", 1, 16384, 3},
    {"zero", 0, 0, 5},
    {"mixed walk", 12345, -23456, 40},
    {"near-max products", INT16_MAX, INT16_MAX, 3},
};

int main(void) {
  policy16 = policy32;
  policy16.width = 16;
  int failed = 0;

  /* The corpus must discriminate. At the saturated rail the true add-half
   * carries across the shift (2^24, floor gives 2^24 - 1), which is exactly
   * the value a lowering that adds half INTO the saturating accumulator
   * cannot produce; the negative half tie separates the two nearest rules. */
  int64_t rail = accumulate(INT16_MIN, INT16_MIN, 512, &policy32);
  if (export_reference(rail, NEAREST_TIES_POSITIVE, SATURATE, &policy32) != (1 << 24) ||
      export_reference(rail, TOWARD_NEGATIVE, SATURATE, &policy32) != (1 << 24) - 1) {
    fprintf(stderr, "rail witness lost the carry discrimination\n");
    failed = 1;
  }
  int64_t tie = accumulate(1, -16384, 3, &policy32);
  if (export_reference(tie, NEAREST_TIES_POSITIVE, SATURATE, &policy32) != -1 ||
      export_reference(tie, NEAREST_EVEN, SATURATE, &policy32) != -2) {
    fprintf(stderr, "negative half tie lost the tie-rule discrimination\n");
    failed = 1;
  }

  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const struct Case *c = &cases[i];
    int32_t expected32 = (int32_t)export_reference(accumulate(c->lhs, c->rhs, c->count, &policy32),
                                                   NEAREST_TIES_POSITIVE, SATURATE, &policy32);
    int32_t actual32 = ondsp_repeat_mac_ntp(c->lhs, c->rhs, c->count);
    if (actual32 != expected32) {
      fprintf(stderr, "%s (i32): got %d, expected %d\n", c->name, actual32, expected32);
      failed = 1;
    }
    int16_t expected16 = (int16_t)export_reference(accumulate(c->lhs, c->rhs, c->count, &policy16),
                                                   NEAREST_TIES_POSITIVE, SATURATE, &policy16);
    int16_t actual16 = ondsp_repeat_mac_ntp_i16(c->lhs, c->rhs, c->count);
    if (actual16 != expected16) {
      fprintf(stderr, "%s (i16): got %d, expected %d\n", c->name, actual16, expected16);
      failed = 1;
    }
  }
  return failed;
}
