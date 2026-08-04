#ifndef ONDRIX_SUPPORT_ELEMENTWISEQ15CONTRACT_H
#define ONDRIX_SUPPORT_ELEMENTWISEQ15CONTRACT_H

#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"

#include <cstdint>

namespace ondrix {

// The exact per-element contract of the unary `ondrix` elementwise members,
// as one definition. A certificate is only worth its name if the arithmetic
// it certifies is the arithmetic the operation declares, so the transform
// that reads it and the tests that check it share this and nothing else.
enum class ElementwiseUnaryKind { Abs, Negate, Offset, Shift };

struct ElementwiseUnaryStep {
  ElementwiseUnaryKind kind;
  // `bias` for Offset, `amount` for Shift, unused otherwise.
  int64_t parameter = 0;
  ondsp::RoundingMode rounding = ondsp::RoundingMode::NearestEven;
  ondsp::OverflowMode overflow = ondsp::OverflowMode::Saturate;
};

// Round `value` right by `shift` bits under the declared rule, stated as a
// floor quotient plus a non-negative remainder so no rule is expressed as a
// bias added in the destination width.
inline int64_t roundSignedShift(int64_t value, unsigned shift, ondsp::RoundingMode rounding) {
  if (shift == 0)
    return value;
  int64_t divisor = int64_t(1) << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  int64_t half = divisor / 2;
  switch (rounding) {
  case ondsp::RoundingMode::TowardNegative:
    break;
  case ondsp::RoundingMode::TowardZero:
    if (value < 0 && remainder != 0)
      ++quotient;
    break;
  case ondsp::RoundingMode::NearestEven:
    if (remainder > half || (remainder == half && (quotient & 1)))
      ++quotient;
    break;
  case ondsp::RoundingMode::NearestTiesPositive:
    if (remainder >= half)
      ++quotient;
    break;
  }
  return quotient;
}

inline int64_t narrowToQ15(int64_t value, ondsp::OverflowMode overflow) {
  if (overflow == ondsp::OverflowMode::Saturate)
    return value > 32767 ? 32767 : (value < -32768 ? -32768 : value);
  int64_t bits = value & 0xFFFF;
  return bits >= 32768 ? bits - 65536 : bits;
}

// One element of one unary member: the exact integer result, then the single
// declared boundary. `value` is a raw signed Q1.15 input.
inline int64_t applyElementwiseUnary(int64_t value, const ElementwiseUnaryStep &step) {
  switch (step.kind) {
  case ElementwiseUnaryKind::Abs:
    return narrowToQ15(value < 0 ? -value : value, step.overflow);
  case ElementwiseUnaryKind::Negate:
    return narrowToQ15(-value, step.overflow);
  case ElementwiseUnaryKind::Offset:
    return narrowToQ15(value + step.parameter, step.overflow);
  case ElementwiseUnaryKind::Shift:
    if (step.parameter >= 0)
      return narrowToQ15(value << step.parameter, step.overflow);
    return narrowToQ15(roundSignedShift(value, unsigned(-step.parameter), step.rounding),
                       step.overflow);
  }
  return value;
}

// Exhaustive equivalence over the whole Q1.15 input domain. Every unary
// member has 65536 possible inputs, so this decides the rewrite rather than
// sampling it, and there is no analogue for a binary member: `mult` has a
// 2^32 domain and neither operand is known at compile time, which is why the
// family's binary members have no fusion certificate at all.
inline bool certifyUnaryChain(const ElementwiseUnaryStep &inner, const ElementwiseUnaryStep &outer,
                              const ElementwiseUnaryStep &merged) {
  for (int64_t value = -32768; value <= 32767; ++value)
    if (applyElementwiseUnary(applyElementwiseUnary(value, inner), outer) !=
        applyElementwiseUnary(value, merged))
      return false;
  return true;
}

inline bool certifyUnaryChainIsIdentity(const ElementwiseUnaryStep &inner,
                                        const ElementwiseUnaryStep &outer) {
  for (int64_t value = -32768; value <= 32767; ++value)
    if (applyElementwiseUnary(applyElementwiseUnary(value, inner), outer) != value)
      return false;
  return true;
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_ELEMENTWISEQ15CONTRACT_H
