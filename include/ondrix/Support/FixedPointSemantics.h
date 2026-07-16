#ifndef ONDRIX_SUPPORT_FIXEDPOINTSEMANTICS_H
#define ONDRIX_SUPPORT_FIXEDPOINTSEMANTICS_H

#include "llvm/ADT/APInt.h"

namespace ondrix::fixedpoint {

enum class AccumulatorUpdateOperation { Add, Subtract };
enum class AccumulatorOverflowMode { Wrap, Saturate };
enum class RoundingMode { TowardNegative, NearestEven, TowardZero };

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

} // namespace ondrix::fixedpoint

#endif
