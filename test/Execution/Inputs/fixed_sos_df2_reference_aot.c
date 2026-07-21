#include <limits.h>
#include <stdint.h>
#include <stdio.h>

typedef int32_t (*Q15Kernel)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                             int32_t, int32_t);
typedef int32_t (*Q31Kernel)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                             int32_t, int32_t);

extern int32_t fixed_sos_df2_q15_sat_update_mixed_export(int32_t, int32_t, int32_t, int32_t,
                                                         int32_t, int32_t, int32_t, int32_t,
                                                         int32_t, int32_t);
extern int32_t fixed_sos_df2_q31_saturate(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                          int32_t, int32_t, int32_t, int32_t);
extern int32_t fixed_sos_df2_q31_wrap(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                      int32_t, int32_t, int32_t);
extern int64_t fixed_sos_df2_q31_sat_update_raw(int32_t, int32_t, int32_t, int32_t, int32_t,
                                                int32_t);

enum OverflowMode { WRAP, SATURATE };
enum RoundingMode { TOWARD_NEGATIVE, NEAREST_EVEN, TOWARD_ZERO };

struct Policy {
  unsigned width;
  unsigned fractional_bits;
  unsigned accumulator_width;
  enum OverflowMode update_overflow;
  enum RoundingMode state_rounding;
  enum OverflowMode state_overflow;
  enum RoundingMode output_rounding;
  enum OverflowMode output_overflow;
};

struct Step {
  int64_t output;
  int64_t d1;
  int64_t d2;
};

static int64_t wrap_signed(__int128 value, unsigned width) {
  __int128 modulus = (__int128)1 << width;
  __int128 bits = value % modulus;
  if (bits < 0)
    bits += modulus;
  if (bits >= modulus / 2)
    bits -= modulus;
  return (int64_t)bits;
}

static int64_t clamp_signed(__int128 value, unsigned width) {
  __int128 minimum = -((__int128)1 << (width - 1));
  __int128 maximum = ((__int128)1 << (width - 1)) - 1;
  if (value < minimum)
    return (int64_t)minimum;
  if (value > maximum)
    return (int64_t)maximum;
  return (int64_t)value;
}

static int64_t update_reference(int64_t accumulator, int64_t lhs, int64_t rhs,
                                const struct Policy *policy) {
  __int128 updated = (__int128)accumulator + (__int128)lhs * (__int128)rhs;
  return policy->update_overflow == WRAP ? wrap_signed(updated, policy->accumulator_width)
                                         : clamp_signed(updated, policy->accumulator_width);
}

static __int128 floor_divide_by_power_of_two(__int128 value, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

static int64_t export_reference(int64_t accumulator, enum RoundingMode rounding,
                                enum OverflowMode overflow, const struct Policy *policy) {
  __int128 quotient = floor_divide_by_power_of_two(accumulator, policy->fractional_bits);
  __int128 divisor = (__int128)1 << policy->fractional_bits;
  __int128 remainder = (__int128)accumulator - quotient * divisor;
  if (rounding == TOWARD_ZERO && accumulator < 0 && remainder != 0)
    ++quotient;
  if (rounding == NEAREST_EVEN) {
    __int128 half = divisor / 2;
    if (remainder > half || (remainder == half && quotient % 2 != 0))
      ++quotient;
  }
  return overflow == WRAP ? wrap_signed(quotient, policy->width)
                          : clamp_signed(quotient, policy->width);
}

static struct Step step_reference(int64_t input, int64_t scale, int64_t b0, int64_t b1, int64_t b2,
                                  int64_t a1, int64_t a2, int64_t d1, int64_t d2,
                                  const struct Policy *policy) {
  int64_t state = update_reference(0, input, scale, policy);
  state = update_reference(state, d1, a1, policy);
  state = update_reference(state, d2, a2, policy);
  int64_t next_d1 = export_reference(state, policy->state_rounding, policy->state_overflow, policy);

  int64_t output_accumulator = update_reference(0, next_d1, b0, policy);
  output_accumulator = update_reference(output_accumulator, d1, b1, policy);
  output_accumulator = update_reference(output_accumulator, d2, b2, policy);
  int64_t output = export_reference(output_accumulator, policy->output_rounding,
                                    policy->output_overflow, policy);
  return (struct Step){output, next_d1, d1};
}

static struct Step invoke_q15(Q15Kernel kernel, int16_t input, int16_t scale, int16_t b0,
                              int16_t b1, int16_t b2, int16_t a1, int16_t a2, int16_t d1,
                              int16_t d2) {
  return (struct Step){kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 0),
                       kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 1),
                       kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 2)};
}

static struct Step invoke_q31(Q31Kernel kernel, int32_t input, int32_t scale, int32_t b0,
                              int32_t b1, int32_t b2, int32_t a1, int32_t a2, int32_t d1,
                              int32_t d2) {
  return (struct Step){kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 0),
                       kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 1),
                       kernel(input, scale, b0, b1, b2, a1, a2, d1, d2, 2)};
}

static int check_step(const char *name, unsigned index, struct Step expected, struct Step actual) {
  if (expected.output == actual.output && expected.d1 == actual.d1 && expected.d2 == actual.d2)
    return 0;
  fprintf(stderr, "%s[%u]: expected (y=%lld,d1=%lld,d2=%lld), got (y=%lld,d1=%lld,d2=%lld)\n", name,
          index, (long long)expected.output, (long long)expected.d1, (long long)expected.d2,
          (long long)actual.output, (long long)actual.d1, (long long)actual.d2);
  return 1;
}

static int check_q15_sequence(void) {
  static const int16_t inputs[] = {1,     3,     -1, -3,  INT16_MAX, INT16_MIN, 12345, -23456,
                                   16384, -8192, 7,  -11, 30000,     -30000,    1024,  -2048};
  const int16_t scale = 24576;
  const int16_t b0 = 16384;
  const int16_t b1 = -8192;
  const int16_t b2 = 4096;
  const int16_t a1 = 8192;
  const int16_t a2 = -4096;
  const struct Policy policy = {16, 15, 40, SATURATE, NEAREST_EVEN, SATURATE, TOWARD_ZERO, WRAP};
  int64_t reference_d1 = 3000;
  int64_t reference_d2 = -5000;
  int16_t object_d1 = (int16_t)reference_d1;
  int16_t object_d2 = (int16_t)reference_d2;
  int failed = 0;

  for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    struct Step expected =
        step_reference(inputs[i], scale, b0, b1, b2, a1, a2, reference_d1, reference_d2, &policy);
    struct Step actual = invoke_q15(fixed_sos_df2_q15_sat_update_mixed_export, inputs[i], scale, b0,
                                    b1, b2, a1, a2, object_d1, object_d2);
    failed |= check_step("q15 sequence", i, expected, actual);
    reference_d1 = expected.d1;
    reference_d2 = expected.d2;
    object_d1 = (int16_t)actual.d1;
    object_d2 = (int16_t)actual.d2;
  }
  return failed;
}

static int check_q15_impulse_convention(void) {
  const int16_t scale = 32767;
  const int16_t b0 = 16384;
  const int16_t b1 = 8192;
  const int16_t b2 = 4096;
  const int16_t a1 = 12288;
  const int16_t a2 = -4096;
  int failed = 0;

  struct Step step0 =
      invoke_q15(fixed_sos_df2_q15_sat_update_mixed_export, 16384, scale, b0, b1, b2, a1, a2, 0, 0);
  failed |= check_step("q15 impulse convention", 0, (struct Step){8192, 16384, 0}, step0);
  struct Step step1 = invoke_q15(fixed_sos_df2_q15_sat_update_mixed_export, 0, scale, b0, b1, b2,
                                 a1, a2, (int16_t)step0.d1, (int16_t)step0.d2);
  failed |= check_step("q15 impulse convention", 1, (struct Step){7168, 6144, 16384}, step1);
  struct Step step2 = invoke_q15(fixed_sos_df2_q15_sat_update_mixed_export, 0, scale, b0, b1, b2,
                                 a1, a2, (int16_t)step1.d1, (int16_t)step1.d2);
  failed |= check_step("q15 impulse convention", 2, (struct Step){3712, 256, 6144}, step2);
  return failed;
}

static int check_q31_update_order(void) {
  __int128 negative_term = (__int128)INT32_MIN * INT32_MAX;
  __int128 expected_wide = (__int128)INT64_MAX + negative_term;
  __int128 deferred_wide =
      (__int128)INT32_MIN * INT32_MIN + (__int128)INT32_MIN * INT32_MIN + negative_term;
  int64_t actual = fixed_sos_df2_q31_sat_update_raw(INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN,
                                                    INT32_MIN, INT32_MAX);
  if (expected_wide == deferred_wide || actual != (int64_t)expected_wide) {
    fprintf(stderr, "Q31 per-update saturation order mismatch\n");
    return 1;
  }
  return 0;
}

static int check_output_overflow(void) {
  int failed = 0;
  struct Step q15 = invoke_q15(fixed_sos_df2_q15_sat_update_mixed_export, INT16_MIN, INT16_MIN,
                               INT16_MAX, INT16_MAX, INT16_MAX, 0, 0, INT16_MAX, INT16_MAX);
  if (q15.output != 32762) {
    fprintf(stderr, "Q15 wrapping output overflow: expected 32762, got %lld\n",
            (long long)q15.output);
    failed = 1;
  }

  struct Step q31 = invoke_q31(fixed_sos_df2_q31_saturate, INT32_MIN, INT32_MIN, INT32_MAX,
                               INT32_MAX, INT32_MAX, 0, 0, INT32_MAX, INT32_MAX);
  if (q31.output != INT32_MAX) {
    fprintf(stderr, "Q31 saturating output overflow: expected %d, got %lld\n", INT32_MAX,
            (long long)q31.output);
    failed = 1;
  }
  return failed;
}

static int check_q31_boundary(void) {
  const struct Policy saturating = {32,       31,           64,      SATURATE, TOWARD_NEGATIVE,
                                    SATURATE, NEAREST_EVEN, SATURATE};
  const struct Policy wrapping = {32, 31, 64, WRAP, TOWARD_ZERO, WRAP, TOWARD_NEGATIVE, WRAP};
  int failed = 0;
  struct Step expected_saturating =
      step_reference(INT32_MIN, INT32_MIN, 0, 0, 0, INT32_MIN, 0, INT32_MIN, 0, &saturating);
  struct Step actual_saturating = invoke_q31(fixed_sos_df2_q31_saturate, INT32_MIN, INT32_MIN, 0, 0,
                                             0, INT32_MIN, 0, INT32_MIN, 0);
  failed |= check_step("q31 saturating 65-bit boundary", 0, expected_saturating, actual_saturating);

  struct Step expected_wrapping =
      step_reference(INT32_MIN, INT32_MIN, 0, 0, 0, INT32_MIN, 0, INT32_MIN, 0, &wrapping);
  struct Step actual_wrapping =
      invoke_q31(fixed_sos_df2_q31_wrap, INT32_MIN, INT32_MIN, 0, 0, 0, INT32_MIN, 0, INT32_MIN, 0);
  failed |= check_step("q31 wrapping 65-bit boundary", 0, expected_wrapping, actual_wrapping);
  if (expected_saturating.d1 != INT32_MAX || expected_wrapping.d1 != 0) {
    fprintf(stderr, "independent Q31 boundary equation has unexpected endpoints\n");
    failed = 1;
  }
  if (actual_saturating.d1 == actual_wrapping.d1) {
    fprintf(stderr, "Q31 update policies did not diverge at the 65-bit boundary\n");
    failed = 1;
  }
  return failed;
}

static int check_q31_sequence(const char *name, Q31Kernel kernel, const struct Policy *policy) {
  static const int32_t inputs[] = {1,          -1,         INT32_MAX,  INT32_MIN,
                                   1073741824, -536870912, 123456789,  -987654321,
                                   3,          -3,         2000000000, -2000000000};
  const int32_t scale = 1610612736;
  const int32_t b0 = 1073741824;
  const int32_t b1 = -536870912;
  const int32_t b2 = 268435456;
  const int32_t a1 = 536870912;
  const int32_t a2 = -268435456;
  int64_t reference_d1 = 123456789;
  int64_t reference_d2 = -345678901;
  int32_t object_d1 = (int32_t)reference_d1;
  int32_t object_d2 = (int32_t)reference_d2;
  int failed = 0;

  for (unsigned i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    struct Step expected =
        step_reference(inputs[i], scale, b0, b1, b2, a1, a2, reference_d1, reference_d2, policy);
    struct Step actual =
        invoke_q31(kernel, inputs[i], scale, b0, b1, b2, a1, a2, object_d1, object_d2);
    failed |= check_step(name, i, expected, actual);
    reference_d1 = expected.d1;
    reference_d2 = expected.d2;
    object_d1 = (int32_t)actual.d1;
    object_d2 = (int32_t)actual.d2;
  }
  return failed;
}

int main(void) {
  const struct Policy q31_saturating = {32,       31,           64,      SATURATE, TOWARD_NEGATIVE,
                                        SATURATE, NEAREST_EVEN, SATURATE};
  const struct Policy q31_wrapping = {32, 31, 64, WRAP, TOWARD_ZERO, WRAP, TOWARD_NEGATIVE, WRAP};
  int failed = 0;
  failed |= check_q15_sequence();
  failed |= check_q15_impulse_convention();
  failed |= check_q31_update_order();
  failed |= check_output_overflow();
  failed |= check_q31_boundary();
  failed |=
      check_q31_sequence("q31 saturating sequence", fixed_sos_df2_q31_saturate, &q31_saturating);
  failed |= check_q31_sequence("q31 wrapping sequence", fixed_sos_df2_q31_wrap, &q31_wrapping);
  return failed;
}
