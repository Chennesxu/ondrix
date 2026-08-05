#ifndef ONDRIX_SUPPORT_GAINQ15CONTRACT_H
#define ONDRIX_SUPPORT_GAINQ15CONTRACT_H

#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"
#include "ondrix/Support/ElementwiseQ15Contract.h"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace ondrix {

// The exact `ondrix.gain` element contract, shared by every transformation
// that reasons about a gain boundary (single-definition discipline as in
// ElementwiseQ15Contract.h).

// The tie rules `ondrix.gain` admits, and therefore the only ones the
// certificates model. Anything else is refused before any arithmetic runs.
inline bool isAdmittedGainRounding(ondsp::RoundingMode mode) {
  return mode == ondsp::RoundingMode::NearestEven ||
         mode == ondsp::RoundingMode::NearestTiesPositive;
}

// One element of `ondrix.gain`: exact integer product, requantization by 15
// under the declared tie rule, i16 saturation. The arithmetic is the shared
// shift primitive; this wrapper adds gain's admission property, so an
// unmodeled mode aborts instead of being silently reinterpreted.
inline int64_t applyGainQ15(int64_t value, int64_t gain, ondsp::RoundingMode mode) {
  if (!isAdmittedGainRounding(mode))
    llvm_unreachable("gain admits only the two nearest tie rules");
  return narrowToQ15(roundSignedShift(value * gain, 15, mode), ondsp::OverflowMode::Saturate);
}

// The Q1.15 quantization of the exact rational product `a * b / 2^15` under a
// declared tie rule. This is pure integer arithmetic (no binary64, no tie
// guard needed): the gain contract itself applied to two constants.
inline int64_t quantizeQ15Product(int64_t lhs, int64_t rhs, ondsp::RoundingMode mode) {
  return applyGainQ15(lhs, rhs, mode);
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_GAINQ15CONTRACT_H
