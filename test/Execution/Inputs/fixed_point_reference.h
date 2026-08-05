#ifndef ONDRIX_TEST_EXECUTION_INPUTS_FIXED_POINT_REFERENCE_H
#define ONDRIX_TEST_EXECUTION_INPUTS_FIXED_POINT_REFERENCE_H

#include <stdint.h>

// Share only fixed-width integer primitives. Each execution test retains an
// independent recurrence implementation and independent hard-coded goldens.
enum OverflowMode { WRAP, SATURATE };
enum RoundingMode { TOWARD_NEGATIVE, NEAREST_EVEN, TOWARD_ZERO, NEAREST_TIES_POSITIVE };

struct Policy {
  unsigned width;
  unsigned frac;
  unsigned accumulator_width;
  unsigned accumulator_frac;
  enum OverflowMode update_overflow;
  enum RoundingMode state_rounding;
  enum OverflowMode state_overflow;
  enum RoundingMode output_rounding;
  enum OverflowMode output_overflow;
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
  unsigned shift = policy->accumulator_frac - policy->frac;
  __int128 quotient = floor_divide_by_power_of_two(accumulator, shift);
  __int128 divisor = (__int128)1 << shift;
  __int128 remainder = (__int128)accumulator - quotient * divisor;
  if (rounding == TOWARD_ZERO && accumulator < 0 && remainder != 0)
    ++quotient;
  if (rounding == NEAREST_EVEN) {
    __int128 half = divisor / 2;
    if (remainder > half || (remainder == half && quotient % 2 != 0))
      ++quotient;
  }
  // ITU-style add-half-then-floor-shift, total in __int128 — deliberately not
  // the compiler's quotient/remainder form.
  if (rounding == NEAREST_TIES_POSITIVE)
    quotient = floor_divide_by_power_of_two((__int128)accumulator + divisor / 2, shift);
  return overflow == WRAP ? wrap_signed(quotient, policy->width)
                          : clamp_signed(quotient, policy->width);
}

#endif
