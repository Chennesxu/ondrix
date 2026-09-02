#ifndef ONDRIX_SUPPORT_GUARDEDFIXEDQUANTIZATION_H
#define ONDRIX_SUPPORT_GUARDEDFIXEDQUANTIZATION_H

#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>

namespace ondrix {

// Admissibility distance from a Q15 rounding half-integer, in LSB units
// (2^-20). A binary64 coefficient estimate is admissible only when it lies at
// least this far from every rounding tie. The guard makes the emitted integer
// conditional on a declared error budget rather than on one host libm: for
// any evaluation chain (libm sin/cos, the binary64 pi constant, angle and
// ratio arithmetic) whose total error stays below the guard — the documented
// budget is more than three orders of magnitude below it — the admissible
// estimate and the real-valued definition provably round to the same integer.
inline constexpr double kQ15TieGuardLsb = 9.5367431640625e-07;

struct GuardedQ15Value {
  int16_t value;
  bool saturated;
};

// One round-half-even signed Q1.15 quantization of a binary64 estimate under
// the tie guard. Returns std::nullopt when the estimate is inadmissible (the
// caller fails closed); reports clamping to [-32768, 32767] so callers can
// count declared saturation. Because admissible ties are unreachable, the
// half-even branch never needs a parity decision.
inline std::optional<GuardedQ15Value> quantizeGuardedQ15(double real) {
  // Shared fail-closed infrastructure must be total on its own: a NaN,
  // infinity, or astronomically out-of-range estimate indicates a broken
  // producer, not declared saturation, and must never reach the
  // floating-point-to-integer conversion below (undefined outside the
  // destination range). Every legitimate design value is orders of
  // magnitude inside the 2^62 bound.
  if (!std::isfinite(real))
    return std::nullopt;
  double scaled = real * 32768.0;
  double lower = std::floor(scaled);
  if (std::fabs(lower) >= 4611686018427387904.0)
    return std::nullopt;
  double fraction = scaled - lower;
  if (std::fabs(fraction - 0.5) < kQ15TieGuardLsb)
    return std::nullopt;
  int64_t quantized = static_cast<int64_t>(lower) + (fraction > 0.5 ? 1 : 0);
  if (quantized > 32767)
    return GuardedQ15Value{32767, true};
  if (quantized < -32768)
    return GuardedQ15Value{-32768, true};
  return GuardedQ15Value{static_cast<int16_t>(quantized), false};
}

// Admissibility distance from a Q31 rounding half-integer, in LSB units
// (2^-13). It is deliberately NOT the Q15 constant: the guard has to clear the
// binary64 representation granularity, and that granularity measured in LSB
// units grows with the format. A scaled coefficient approaching 2^31 carries
// an ulp of 2^-21 LSB, so the Q15 guard of 2^-20 would stand one single bit
// above it and prove nothing; 2^-13 clears it by 256. The bound is far below
// what the coefficients actually need: the worst type-II DCT tie margin over
// all 5456 coefficients of the supported extents is 4.93e-03 LSB, about 40
// times this guard, measured offline at 50 decimal digits.
inline constexpr double kQ31TieGuardLsb = 1.220703125e-04;

struct GuardedQ31Value {
  int32_t value;
  bool saturated;
};

// The Q31 counterpart of quantizeGuardedQ15, same fail-closed contract.
inline std::optional<GuardedQ31Value> quantizeGuardedQ31(double real) {
  if (!std::isfinite(real))
    return std::nullopt;
  double scaled = real * 2147483648.0;
  double lower = std::floor(scaled);
  if (std::fabs(lower) >= 4611686018427387904.0)
    return std::nullopt;
  double fraction = scaled - lower;
  if (std::fabs(fraction - 0.5) < kQ31TieGuardLsb)
    return std::nullopt;
  int64_t quantized = static_cast<int64_t>(lower) + (fraction > 0.5 ? 1 : 0);
  if (quantized > INT32_MAX)
    return GuardedQ31Value{INT32_MAX, true};
  if (quantized < INT32_MIN)
    return GuardedQ31Value{INT32_MIN, true};
  return GuardedQ31Value{static_cast<int32_t>(quantized), false};
}

// The same guarded nearest-integer quantization for a table whose scale is
// not a Q-format's own: `real * scale` under the Q31 tie guard, with no clamp,
// so a caller states the range its table occupies and checks it. The Q31
// guard is the right one for every scale between 2^30 and 2^33, where the
// binary64 ulp measured in LSB lies between 2^-22 and 2^-19.
inline std::optional<int64_t> quantizeGuardedAtScale(double real, double scale) {
  if (!std::isfinite(real))
    return std::nullopt;
  double scaled = real * scale;
  double lower = std::floor(scaled);
  if (std::fabs(lower) >= 4611686018427387904.0)
    return std::nullopt;
  double fraction = scaled - lower;
  if (std::fabs(fraction - 0.5) < kQ31TieGuardLsb)
    return std::nullopt;
  return static_cast<int64_t>(lower) + (fraction > 0.5 ? 1 : 0);
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_GUARDEDFIXEDQUANTIZATION_H
