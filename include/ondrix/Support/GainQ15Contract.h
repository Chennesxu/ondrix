#ifndef ONDRIX_SUPPORT_GAINQ15CONTRACT_H
#define ONDRIX_SUPPORT_GAINQ15CONTRACT_H

#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace ondrix {

// The exact `ondrix.gain` element contract, shared by every transformation
// that reasons about a gain boundary. There is exactly one definition: a
// certificate is only worth its name if the arithmetic it certifies is the
// arithmetic the operation declares, so passes must not carry private copies
// that can drift apart.
//
// The tie rules `ondrix.gain` admits, and therefore the only ones the
// certificates model. Anything else is refused before any arithmetic runs.
inline bool isAdmittedGainRounding(ondsp::RoundingMode mode) {
  return mode == ondsp::RoundingMode::NearestEven ||
         mode == ondsp::RoundingMode::NearestTiesPositive;
}

// One element of `ondrix.gain`: exact integer product, requantization by 15 in
// explicit floor-division form under the declared tie rule, i16 saturation.
// The operation admits two tie rules and they are not interchangeable — a
// certificate is only valid for the rule it was evaluated under.
inline int64_t applyGainQ15(int64_t value, int64_t gain, ondsp::RoundingMode mode) {
  int64_t product = value * gain;
  int64_t quotient = product / 32768;
  int64_t remainder = product % 32768;
  if (remainder < 0) {
    --quotient;
    remainder += 32768;
  }
  switch (mode) {
  case ondsp::RoundingMode::NearestTiesPositive:
    // Ties toward +infinity: every remainder of at least half steps up.
    if (remainder >= 16384)
      ++quotient;
    break;
  case ondsp::RoundingMode::NearestEven:
    if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
      ++quotient;
    break;
  case ondsp::RoundingMode::TowardNegative:
  case ondsp::RoundingMode::TowardZero:
    // Unreachable: `isAdmittedGainRounding` gates every caller, so an
    // unmodeled mode must abort rather than be silently reinterpreted as
    // nearest_even and certify a rewrite the program never asked for.
    llvm_unreachable("gain admits only the two nearest tie rules");
  }
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return quotient;
}

// The Q1.15 quantization of the exact rational product `a * b / 2^15` under a
// declared tie rule. This is pure integer arithmetic (no binary64, no tie
// guard needed): the gain contract itself applied to two constants.
inline int64_t quantizeQ15Product(int64_t lhs, int64_t rhs, ondsp::RoundingMode mode) {
  return applyGainQ15(lhs, rhs, mode);
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_GAINQ15CONTRACT_H
