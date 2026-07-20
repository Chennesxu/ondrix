#ifndef ONDRIX_SUPPORT_FIXEDPOINTSEMANTICS_H
#define ONDRIX_SUPPORT_FIXEDPOINTSEMANTICS_H

#include "llvm/ADT/APInt.h"

#include <optional>

namespace ondrix::fixedpoint {

enum class AccumulatorUpdateOperation { Add, Subtract };
enum class AccumulatorOverflowMode { Wrap, Saturate };
enum class RoundingMode { TowardNegative, NearestEven, TowardZero };

/// Numeric policy for one signed fixed-point direct-form-II SOS section.
/// Samples, coefficients, scale, state, and output use the same storage width
/// and fractional position. Products and accumulator updates use twice that
/// fractional position.
struct SignedUniformSosDf2Policy {
  unsigned storageWidth;
  unsigned fractionalBits;
  unsigned accumulatorWidth;
  AccumulatorOverflowMode updateOverflow;
  RoundingMode stateRounding;
  AccumulatorOverflowMode stateOverflow;
  RoundingMode outputRounding;
  AccumulatorOverflowMode outputOverflow;
};

struct SignedUniformSosDf2Step {
  llvm::APInt output;
  llvm::APInt d1;
  llvm::APInt d2;
};

/// Returns the exact intermediate width required by an accumulator update.
unsigned getAccumulatorUpdateIntermediateWidth(unsigned accumulatorWidth, unsigned productWidth);

/// Computes the exact signed full product in lhs.width + rhs.width bits.
llvm::APInt computeSignedFullProduct(const llvm::APInt &lhs, const llvm::APInt &rhs);

/// Returns trunc_W(ashr(P, W)) for equal-width signed operands and their exact
/// signed 2W-bit product P.
llvm::APInt computeSignedRawHighProduct(const llvm::APInt &lhs, const llvm::APInt &rhs);

/// Applies one signed accumulator update and returns accumulator-width raw bits.
llvm::APInt updateSignedAccumulator(const llvm::APInt &accumulator, const llvm::APInt &product,
                                    AccumulatorUpdateOperation operation,
                                    AccumulatorOverflowMode overflowMode);

/// Computes an exact signed full product and applies one accumulator update.
llvm::APInt multiplyAccumulateSigned(const llvm::APInt &accumulator, const llvm::APInt &lhs,
                                     const llvm::APInt &rhs, AccumulatorUpdateOperation operation,
                                     AccumulatorOverflowMode overflowMode);

/// Discards fractional bits with explicit rounding, then applies destination
/// signed overflow handling and returns destination-width raw bits.
llvm::APInt exportSignedAccumulator(const llvm::APInt &accumulator,
                                    unsigned fractionalBitsToDiscard, unsigned destinationWidth,
                                    RoundingMode roundingMode,
                                    AccumulatorOverflowMode overflowMode);

/// Evaluates one signed, uniform-Q-format direct-form-II SOS section.
/// Coefficients use the order b0, b1, b2, a1, a2. Feedback is additive; a
/// subtractive denominator convention supplies negated a1/a2 values.
///
/// The exact raw-bit sequence is:
///   stateAcc = mac(mac(mac(0, input, scale), d1, a1), d2, a2)
///   nextD1 = export(stateAcc, state policy)
///   outputAcc = mac(mac(mac(0, nextD1, b0), d1, b1), d2, b2)
///   output = export(outputAcc, output policy)
///   nextD2 = d1
///
/// Returns std::nullopt unless the policy and every raw input width agree.
std::optional<SignedUniformSosDf2Step> evaluateSignedUniformSosDf2Section(
    const llvm::APInt &input, const llvm::APInt &scale, const llvm::APInt &b0,
    const llvm::APInt &b1, const llvm::APInt &b2, const llvm::APInt &a1, const llvm::APInt &a2,
    const llvm::APInt &d1, const llvm::APInt &d2, const SignedUniformSosDf2Policy &policy);

} // namespace ondrix::fixedpoint

#endif
