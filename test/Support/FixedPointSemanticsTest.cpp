#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>

using ondrix::fixedpoint::AccumulatorOverflowMode;
using ondrix::fixedpoint::AccumulatorUpdateOperation;
using ondrix::fixedpoint::computeSignedFullProduct;
using ondrix::fixedpoint::exportSignedAccumulator;
using ondrix::fixedpoint::getAccumulatorUpdateIntermediateWidth;
using ondrix::fixedpoint::multiplyAccumulateSigned;
using ondrix::fixedpoint::RoundingMode;

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

bool testAccumulatorIntermediateWidth() {
  return getAccumulatorUpdateIntermediateWidth(40, 32) == 41 &&
         getAccumulatorUpdateIntermediateWidth(16, 32) == 33;
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

bool testAccumulatorExportRounding() {
  bool passed = true;
  passed &=
      expectEqual("floor positive",
                  exportSignedAccumulator(signedValue(8, 5), 2, 8, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Wrap),
                  signedValue(8, 1));
  passed &=
      expectEqual("floor negative",
                  exportSignedAccumulator(signedValue(8, -5), 2, 8, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Wrap),
                  signedValue(8, -2));
  passed &= expectEqual("zero negative",
                        exportSignedAccumulator(signedValue(8, -5), 2, 8, RoundingMode::TowardZero,
                                                AccumulatorOverflowMode::Wrap),
                        signedValue(8, -1));
  passed &= expectEqual("nearest odd tie",
                        exportSignedAccumulator(signedValue(8, 6), 2, 8, RoundingMode::NearestEven,
                                                AccumulatorOverflowMode::Wrap),
                        signedValue(8, 2));
  passed &= expectEqual("nearest even tie",
                        exportSignedAccumulator(signedValue(8, 10), 2, 8, RoundingMode::NearestEven,
                                                AccumulatorOverflowMode::Wrap),
                        signedValue(8, 2));
  passed &= expectEqual("nearest negative odd tie",
                        exportSignedAccumulator(signedValue(8, -6), 2, 8, RoundingMode::NearestEven,
                                                AccumulatorOverflowMode::Wrap),
                        signedValue(8, -2));
  passed &=
      expectEqual("nearest negative even tie",
                  exportSignedAccumulator(signedValue(8, -10), 2, 8, RoundingMode::NearestEven,
                                          AccumulatorOverflowMode::Wrap),
                  signedValue(8, -2));
  return passed;
}

bool testAccumulatorExportOverflow() {
  constexpr unsigned accumulatorWidth = 40;
  constexpr unsigned shift = 15;
  llvm::APInt positiveOverflow = signedValue(accumulatorWidth, int64_t{32768} << shift);
  llvm::APInt negativeOverflow =
      signedValue(accumulatorWidth, int64_t{-32769} * (int64_t{1} << shift));

  bool passed = true;
  passed &=
      expectEqual("export positive saturation",
                  exportSignedAccumulator(positiveOverflow, shift, 16, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Saturate),
                  signedValue(16, 32767));
  passed &=
      expectEqual("export negative saturation",
                  exportSignedAccumulator(negativeOverflow, shift, 16, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Saturate),
                  signedValue(16, -32768));
  passed &=
      expectEqual("export positive wrap",
                  exportSignedAccumulator(positiveOverflow, shift, 16, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Wrap),
                  signedValue(16, -32768));
  passed &=
      expectEqual("export negative wrap",
                  exportSignedAccumulator(negativeOverflow, shift, 16, RoundingMode::TowardNegative,
                                          AccumulatorOverflowMode::Wrap),
                  signedValue(16, 32767));
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

int64_t floorDivideByPowerOfTwo(int64_t value, unsigned shift) {
  int64_t divisor = int64_t{1} << shift;
  int64_t quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

int64_t roundReference(int64_t value, unsigned shift, RoundingMode mode) {
  int64_t quotient = floorDivideByPowerOfTwo(value, shift);
  int64_t divisor = int64_t{1} << shift;
  int64_t remainder = value - quotient * divisor;
  switch (mode) {
  case RoundingMode::TowardNegative:
    return quotient;
  case RoundingMode::TowardZero:
    return value < 0 && remainder != 0 ? quotient + 1 : quotient;
  case RoundingMode::NearestEven:
    return remainder > divisor / 2 || (remainder == divisor / 2 && quotient % 2 != 0) ? quotient + 1
                                                                                      : quotient;
  }
  return quotient;
}

bool testSmallWidthExportExhaustive() {
  constexpr unsigned accumulatorWidth = 5;
  constexpr unsigned destinationWidth = 3;
  constexpr int64_t destinationMinimum = -4;
  constexpr int64_t destinationMaximum = 3;

  for (int64_t value = -16; value <= 15; ++value) {
    for (unsigned shift : {1U, 2U}) {
      for (RoundingMode rounding :
           {RoundingMode::TowardNegative, RoundingMode::NearestEven, RoundingMode::TowardZero}) {
        int64_t rounded = roundReference(value, shift, rounding);
        for (AccumulatorOverflowMode overflowMode :
             {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
          int64_t expected = overflowMode == AccumulatorOverflowMode::Wrap
                                 ? wrapSigned(rounded, destinationWidth)
                                 : std::clamp(rounded, destinationMinimum, destinationMaximum);
          llvm::APInt actual = exportSignedAccumulator(signedValue(accumulatorWidth, value), shift,
                                                       destinationWidth, rounding, overflowMode);
          if (!expectEqual("small-width exhaustive export", actual,
                           signedValue(destinationWidth, expected)))
            return false;
        }
      }
    }
  }
  return true;
}

} // namespace

int main() {
  bool passed = testQ15FullProducts();
  passed &= testAccumulatorIntermediateWidth();
  passed &= testAccumulatorBoundaries();
  passed &= testAccumulatorChains();
  passed &= testAccumulatorExportRounding();
  passed &= testAccumulatorExportOverflow();
  passed &= testSmallWidthExhaustive();
  passed &= testSmallWidthExportExhaustive();
  if (!passed)
    return 1;
  llvm::outs() << "fixed-point accumulator semantics: PASS\n";
  return 0;
}
