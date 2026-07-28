#ifndef ONDRIX_SUPPORT_GUARDEDQ15QUANTIZATION_H
#define ONDRIX_SUPPORT_GUARDEDQ15QUANTIZATION_H

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
  double scaled = real * 32768.0;
  double lower = std::floor(scaled);
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

} // namespace ondrix

#endif // ONDRIX_SUPPORT_GUARDEDQ15QUANTIZATION_H
