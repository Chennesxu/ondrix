#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace ondrix::fixedpoint {

namespace {

bool isValidOverflowMode(AccumulatorOverflowMode mode) {
  switch (mode) {
  case AccumulatorOverflowMode::Wrap:
  case AccumulatorOverflowMode::Saturate:
    return true;
  }
  return false;
}

bool isValidRoundingMode(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::TowardNegative:
  case RoundingMode::NearestTiesPositive:
  case RoundingMode::NearestEven:
  case RoundingMode::TowardZero:
    return true;
  }
  return false;
}

} // namespace

unsigned getAccumulatorUpdateIntermediateWidth(unsigned accumulatorWidth, unsigned productWidth) {
  return std::max(accumulatorWidth, productWidth) + 1;
}

llvm::APInt computeSignedFullProduct(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  unsigned productWidth = lhs.getBitWidth() + rhs.getBitWidth();
  return lhs.sext(productWidth) * rhs.sext(productWidth);
}

llvm::APInt computeSignedRawHighProduct(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  assert(lhs.getBitWidth() == rhs.getBitWidth() &&
         "raw high product requires equal-width operands");
  unsigned operandWidth = lhs.getBitWidth();
  return computeSignedFullProduct(lhs, rhs).ashr(operandWidth).trunc(operandWidth);
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
    case RoundingMode::NearestTiesPositive: {
      // Ties toward +infinity on top of the floor already taken above: a
      // remainder of at least half always steps up. Expressed on the
      // remainder rather than as an in-width add of half, which would
      // overflow near the accumulator maximum.
      llvm::APInt half = one.shl(fractionalBitsToDiscard - 1);
      if (remainder.uge(half))
        rounded += one;
      break;
    }
    case RoundingMode::NearestEven: {
      llvm::APInt half = one.shl(fractionalBitsToDiscard - 1);
      if (remainder.ugt(half) || (remainder == half && rounded[0]))
        rounded += one;
      break;
    }
    }
  }

  switch (overflowMode) {
  case AccumulatorOverflowMode::Wrap:
    return rounded.sextOrTrunc(destinationWidth);
  case AccumulatorOverflowMode::Saturate: {
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
  }
  llvm_unreachable("unhandled declared overflow mode");
}

std::optional<SignedUniformSosDf2Step> evaluateSignedUniformSosDf2Section(
    const llvm::APInt &input, const llvm::APInt &scale, const llvm::APInt &b0,
    const llvm::APInt &b1, const llvm::APInt &b2, const llvm::APInt &a1, const llvm::APInt &a2,
    const llvm::APInt &d1, const llvm::APInt &d2, const SignedUniformSosDf2Policy &policy) {
  if (policy.storageWidth == 0 || policy.fractionalBits > policy.storageWidth ||
      policy.accumulatorWidth < policy.storageWidth ||
      policy.accumulatorWidth - policy.storageWidth < policy.storageWidth ||
      !isValidOverflowMode(policy.updateOverflow) || !isValidRoundingMode(policy.stateRounding) ||
      !isValidOverflowMode(policy.stateOverflow) || !isValidRoundingMode(policy.outputRounding) ||
      !isValidOverflowMode(policy.outputOverflow))
    return std::nullopt;

  auto hasStorageWidth = [&](const llvm::APInt &value) {
    return value.getBitWidth() == policy.storageWidth;
  };
  if (!hasStorageWidth(input) || !hasStorageWidth(scale) || !hasStorageWidth(b0) ||
      !hasStorageWidth(b1) || !hasStorageWidth(b2) || !hasStorageWidth(a1) ||
      !hasStorageWidth(a2) || !hasStorageWidth(d1) || !hasStorageWidth(d2))
    return std::nullopt;

  auto update = [&](const llvm::APInt &accumulator, const llvm::APInt &lhs,
                    const llvm::APInt &rhs) {
    return multiplyAccumulateSigned(accumulator, lhs, rhs, AccumulatorUpdateOperation::Add,
                                    policy.updateOverflow);
  };

  llvm::APInt stateAccumulator(policy.accumulatorWidth, 0);
  stateAccumulator = update(stateAccumulator, input, scale);
  stateAccumulator = update(stateAccumulator, d1, a1);
  stateAccumulator = update(stateAccumulator, d2, a2);
  llvm::APInt nextD1 =
      exportSignedAccumulator(stateAccumulator, policy.fractionalBits, policy.storageWidth,
                              policy.stateRounding, policy.stateOverflow);

  llvm::APInt outputAccumulator(policy.accumulatorWidth, 0);
  outputAccumulator = update(outputAccumulator, nextD1, b0);
  outputAccumulator = update(outputAccumulator, d1, b1);
  outputAccumulator = update(outputAccumulator, d2, b2);
  llvm::APInt output =
      exportSignedAccumulator(outputAccumulator, policy.fractionalBits, policy.storageWidth,
                              policy.outputRounding, policy.outputOverflow);
  return SignedUniformSosDf2Step{std::move(output), std::move(nextD1), d1};
}

} // namespace ondrix::fixedpoint
