#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>

namespace ondrix::fixedpoint {

llvm::APInt computeSignedFullProduct(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  unsigned productWidth = lhs.getBitWidth() + rhs.getBitWidth();
  return lhs.sext(productWidth) * rhs.sext(productWidth);
}

llvm::APInt updateSignedAccumulator(const llvm::APInt &accumulator, const llvm::APInt &product,
                                    AccumulatorUpdateOperation operation,
                                    AccumulatorOverflowMode overflowMode) {
  unsigned accumulatorWidth = accumulator.getBitWidth();
  unsigned intermediateWidth = std::max(accumulatorWidth, product.getBitWidth()) + 1;
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

} // namespace ondrix::fixedpoint
