#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>

using ondrix::fixedpoint::AccumulatorOverflowMode;
using ondrix::fixedpoint::AccumulatorUpdateOperation;
using ondrix::fixedpoint::computeSignedFullProduct;
using ondrix::fixedpoint::multiplyAccumulateSigned;

namespace {

llvm::APInt signedValue(unsigned width, int64_t value) {
  return llvm::APInt(width, static_cast<uint64_t>(value), true);
}

bool expectEqual(llvm::StringRef name, const llvm::APInt &actual, const llvm::APInt &expected) {
  if (actual == expected)
    return true;
  llvm::SmallString<32> actualText;
  llvm::SmallString<32> expectedText;
  actual.toStringSigned(actualText);
  expected.toStringSigned(expectedText);
  llvm::errs() << name << ": expected " << expectedText << ", got " << actualText << '\n';
  return false;
}

bool testQ15FullProducts() {
  bool passed = true;
  passed &= expectEqual("max times max",
                        computeSignedFullProduct(signedValue(16, 32767), signedValue(16, 32767)),
                        signedValue(32, 1073676289));
  passed &= expectEqual("min times min",
                        computeSignedFullProduct(signedValue(16, -32768), signedValue(16, -32768)),
                        signedValue(32, 1073741824));
  passed &= expectEqual("min times max",
                        computeSignedFullProduct(signedValue(16, -32768), signedValue(16, 32767)),
                        signedValue(32, -1073709056));
  return passed;
}

bool testAccumulatorBoundaries() {
  constexpr unsigned accumulatorWidth = 40;
  llvm::APInt maximum = llvm::APInt::getSignedMaxValue(accumulatorWidth);
  llvm::APInt minimum = llvm::APInt::getSignedMinValue(accumulatorWidth);
  llvm::APInt one = signedValue(16, 1);
  llvm::APInt negativeOne = signedValue(16, -1);

  bool passed = true;
  passed &=
      expectEqual("add exact positive boundary",
                  multiplyAccumulateSigned(maximum - 1, one, one, AccumulatorUpdateOperation::Add,
                                           AccumulatorOverflowMode::Saturate),
                  maximum);
  passed &= expectEqual("add positive saturation",
                        multiplyAccumulateSigned(maximum, one, one, AccumulatorUpdateOperation::Add,
                                                 AccumulatorOverflowMode::Saturate),
                        maximum);
  passed &= expectEqual("add negative saturation",
                        multiplyAccumulateSigned(minimum, negativeOne, one,
                                                 AccumulatorUpdateOperation::Add,
                                                 AccumulatorOverflowMode::Saturate),
                        minimum);
  passed &= expectEqual("add exact negative boundary",
                        multiplyAccumulateSigned(minimum + 1, negativeOne, one,
                                                 AccumulatorUpdateOperation::Add,
                                                 AccumulatorOverflowMode::Saturate),
                        minimum);
  passed &= expectEqual("add positive wrap",
                        multiplyAccumulateSigned(maximum, one, one, AccumulatorUpdateOperation::Add,
                                                 AccumulatorOverflowMode::Wrap),
                        minimum);
  passed &= expectEqual("add negative wrap",
                        multiplyAccumulateSigned(minimum, negativeOne, one,
                                                 AccumulatorUpdateOperation::Add,
                                                 AccumulatorOverflowMode::Wrap),
                        maximum);
  passed &= expectEqual("subtract positive saturation",
                        multiplyAccumulateSigned(maximum, negativeOne, one,
                                                 AccumulatorUpdateOperation::Subtract,
                                                 AccumulatorOverflowMode::Saturate),
                        maximum);
  passed &=
      expectEqual("subtract negative saturation",
                  multiplyAccumulateSigned(minimum, one, one, AccumulatorUpdateOperation::Subtract,
                                           AccumulatorOverflowMode::Saturate),
                  minimum);
  passed &= expectEqual("subtract positive wrap",
                        multiplyAccumulateSigned(maximum, negativeOne, one,
                                                 AccumulatorUpdateOperation::Subtract,
                                                 AccumulatorOverflowMode::Wrap),
                        minimum);
  passed &=
      expectEqual("subtract negative wrap",
                  multiplyAccumulateSigned(minimum, one, one, AccumulatorUpdateOperation::Subtract,
                                           AccumulatorOverflowMode::Wrap),
                  maximum);
  return passed;
}

bool testAccumulatorChains() {
  constexpr unsigned accumulatorWidth = 40;
  llvm::APInt maximum = llvm::APInt::getSignedMaxValue(accumulatorWidth);
  llvm::APInt minimum = llvm::APInt::getSignedMinValue(accumulatorWidth);
  llvm::APInt one = signedValue(16, 1);

  llvm::APInt saturating = maximum - 2;
  llvm::APInt wrapping = maximum - 2;
  for (unsigned i = 0; i != 3; ++i) {
    saturating = multiplyAccumulateSigned(saturating, one, one, AccumulatorUpdateOperation::Add,
                                          AccumulatorOverflowMode::Saturate);
    wrapping = multiplyAccumulateSigned(wrapping, one, one, AccumulatorUpdateOperation::Add,
                                        AccumulatorOverflowMode::Wrap);
  }

  bool passed = true;
  passed &= expectEqual("saturating chain", saturating, maximum);
  passed &= expectEqual("wrapping chain", wrapping, minimum);
  return passed;
}

int64_t wrapSigned(int64_t value, unsigned width) {
  int64_t modulus = int64_t{1} << width;
  int64_t bits = value & (modulus - 1);
  return bits >= modulus / 2 ? bits - modulus : bits;
}

bool testSmallWidthExhaustive() {
  constexpr unsigned accumulatorWidth = 4;
  constexpr unsigned operandWidth = 3;
  constexpr int64_t accumulatorMinimum = -8;
  constexpr int64_t accumulatorMaximum = 7;
  constexpr int64_t operandMinimum = -4;
  constexpr int64_t operandMaximum = 3;

  for (int64_t accumulator = accumulatorMinimum; accumulator <= accumulatorMaximum; ++accumulator) {
    for (int64_t lhs = operandMinimum; lhs <= operandMaximum; ++lhs) {
      for (int64_t rhs = operandMinimum; rhs <= operandMaximum; ++rhs) {
        int64_t product = lhs * rhs;
        for (AccumulatorUpdateOperation operation :
             {AccumulatorUpdateOperation::Add, AccumulatorUpdateOperation::Subtract}) {
          int64_t updated = operation == AccumulatorUpdateOperation::Add ? accumulator + product
                                                                         : accumulator - product;
          for (AccumulatorOverflowMode overflowMode :
               {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
            int64_t expected = overflowMode == AccumulatorOverflowMode::Wrap
                                   ? wrapSigned(updated, accumulatorWidth)
                                   : std::clamp(updated, accumulatorMinimum, accumulatorMaximum);
            llvm::APInt actual = multiplyAccumulateSigned(
                signedValue(accumulatorWidth, accumulator), signedValue(operandWidth, lhs),
                signedValue(operandWidth, rhs), operation, overflowMode);
            if (!expectEqual("small-width exhaustive update", actual,
                             signedValue(accumulatorWidth, expected)))
              return false;
          }
        }
      }
    }
  }
  return true;
}

} // namespace

int main() {
  bool passed = testQ15FullProducts();
  passed &= testAccumulatorBoundaries();
  passed &= testAccumulatorChains();
  passed &= testSmallWidthExhaustive();
  if (!passed)
    return 1;
  llvm::outs() << "fixed-point accumulator semantics: PASS\n";
  return 0;
}
