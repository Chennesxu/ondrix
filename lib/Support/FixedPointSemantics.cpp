#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>

namespace ondrix::fixedpoint {

unsigned getAccumulatorUpdateIntermediateWidth(unsigned accumulatorWidth, unsigned productWidth) {
  return std::max(accumulatorWidth, productWidth) + 1;
}

llvm::APInt computeSignedFullProduct(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  unsigned productWidth = lhs.getBitWidth() + rhs.getBitWidth();
  return lhs.sext(productWidth) * rhs.sext(productWidth);
}

llvm::APInt updateSignedAccumulator(const llvm::APInt &accumulator, const llvm::APInt &product,
                                    AccumulatorUpdateOperation operation,
                                    AccumulatorOverflowMode overflowMode) {
  unsigned accumulatorWidth = accumulator.getBitWidth();
  unsigned intermediateWidth =
      getAccumulatorUpdateIntermediateWidth(accumulatorWidth, product.getBitWidth());
  llvm::APInt extendedAccumulator = accumulator.sext(intermediateWidth);
  llvm::APInt extendedProduct = product.sext(intermediateWidth);

  llvm::APInt updated = extendedAccumulator;
  switch (operation) {
  case AccumulatorUpdateOperation::Add:
    updated += extendedProduct;
    break;
  case AccumulatorUpdateOperation::Subtract:
    updated -= extendedProduct;
    break;
  }

  switch (overflowMode) {
  case AccumulatorOverflowMode::Wrap:
    return updated.trunc(accumulatorWidth);
  case AccumulatorOverflowMode::Saturate:
    llvm::APInt minimum = llvm::APInt::getSignedMinValue(accumulatorWidth);
    llvm::APInt maximum = llvm::APInt::getSignedMaxValue(accumulatorWidth);
    if (updated.slt(minimum.sext(intermediateWidth)))
      return minimum;
    if (updated.sgt(maximum.sext(intermediateWidth)))
      return maximum;
    return updated.trunc(accumulatorWidth);
  }

  llvm_unreachable("unknown accumulator overflow mode");
}

llvm::APInt multiplyAccumulateSigned(const llvm::APInt &accumulator, const llvm::APInt &lhs,
                                     const llvm::APInt &rhs, AccumulatorUpdateOperation operation,
                                     AccumulatorOverflowMode overflowMode) {
  return updateSignedAccumulator(accumulator, computeSignedFullProduct(lhs, rhs), operation,
                                 overflowMode);
}

llvm::APInt exportSignedAccumulator(const llvm::APInt &accumulator,
                                    unsigned fractionalBitsToDiscard, unsigned destinationWidth,
                                    RoundingMode roundingMode,
                                    AccumulatorOverflowMode overflowMode) {
  unsigned accumulatorWidth = accumulator.getBitWidth();
  assert(destinationWidth != 0 && "destination width must be nonzero");
  assert(fractionalBitsToDiscard <= accumulatorWidth &&
         "fractional shift must not exceed accumulator width");

  llvm::APInt rounded = accumulator;
  if (fractionalBitsToDiscard != 0) {
    rounded = fractionalBitsToDiscard == accumulatorWidth
                  ? (accumulator.isNegative() ? llvm::APInt::getAllOnes(accumulatorWidth)
                                              : llvm::APInt(accumulatorWidth, 0))
                  : accumulator.ashr(fractionalBitsToDiscard);
    llvm::APInt remainder = fractionalBitsToDiscard == accumulatorWidth
                                ? accumulator
                                : accumulator.trunc(fractionalBitsToDiscard).zext(accumulatorWidth);
    llvm::APInt one(accumulatorWidth, 1);
    switch (roundingMode) {
    case RoundingMode::TowardNegative:
      break;
    case RoundingMode::TowardZero:
      if (accumulator.isNegative() && !remainder.isZero())
        rounded += one;
      break;
    case RoundingMode::NearestEven: {
      llvm::APInt half = one.shl(fractionalBitsToDiscard - 1);
      if (remainder.ugt(half) || (remainder == half && rounded[0]))
        rounded += one;
      break;
    }
    }
  }

  if (overflowMode == AccumulatorOverflowMode::Wrap)
    return rounded.sextOrTrunc(destinationWidth);

  unsigned comparisonWidth = std::max(accumulatorWidth, destinationWidth) + 1;
  llvm::APInt extended = rounded.sext(comparisonWidth);
  llvm::APInt minimum = llvm::APInt::getSignedMinValue(destinationWidth);
  llvm::APInt maximum = llvm::APInt::getSignedMaxValue(destinationWidth);
  if (extended.slt(minimum.sext(comparisonWidth)))
    return minimum;
  if (extended.sgt(maximum.sext(comparisonWidth)))
    return maximum;
  return rounded.sextOrTrunc(destinationWidth);
}

} // namespace ondrix::fixedpoint
