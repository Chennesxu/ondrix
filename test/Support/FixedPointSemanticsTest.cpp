#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>

using ondrix::fixedpoint::AccumulatorOverflowMode;
using ondrix::fixedpoint::AccumulatorUpdateOperation;
using ondrix::fixedpoint::computeSignedFullProduct;
using ondrix::fixedpoint::computeSignedRawHighProduct;
using ondrix::fixedpoint::evaluateSignedUniformSosDf2Section;
using ondrix::fixedpoint::exportSignedAccumulator;
using ondrix::fixedpoint::getAccumulatorUpdateIntermediateWidth;
using ondrix::fixedpoint::multiplyAccumulateSigned;
using ondrix::fixedpoint::RoundingMode;
using ondrix::fixedpoint::SignedUniformSosDf2Policy;

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

bool testCanonicalQ15TwiddleIdentities() {
  const llvm::APInt one = signedValue(16, 32767);
  const llvm::APInt negativeOne = signedValue(16, -32768);
  for (int64_t value = INT16_MIN; value <= INT16_MAX; ++value) {
    llvm::APInt input = signedValue(16, value);
    llvm::APInt genericOne =
        exportSignedAccumulator(computeSignedFullProduct(input, one), 15, 16,
                                RoundingMode::NearestEven, AccumulatorOverflowMode::Saturate);
    int64_t specializedOne = value + (value < -16384 ? 1 : 0) - (value > 16384 ? 1 : 0);
    if (!expectEqual("canonical Q15 +1", genericOne, signedValue(16, specializedOne)))
      return false;

    llvm::APInt genericNegate =
        exportSignedAccumulator(computeSignedFullProduct(input, negativeOne), 15, 16,
                                RoundingMode::NearestEven, AccumulatorOverflowMode::Saturate);
    int64_t specializedNegate = value == INT16_MIN ? INT16_MAX : -value;
    if (!expectEqual("canonical Q15 saturating negate", genericNegate,
                     signedValue(16, specializedNegate)))
      return false;
  }
  return true;
}

bool testQ31FullProducts() {
  bool passed = true;
  passed &= expectEqual(
      "q31 max times max full",
      computeSignedFullProduct(signedValue(32, 2147483647), signedValue(32, 2147483647)),
      signedValue(64, INT64_C(4611686014132420609)));
  passed &= expectEqual(
      "q31 min times min full",
      computeSignedFullProduct(signedValue(32, -2147483648LL), signedValue(32, -2147483648LL)),
      signedValue(64, INT64_C(4611686018427387904)));
  passed &= expectEqual(
      "q31 min times max full",
      computeSignedFullProduct(signedValue(32, -2147483648LL), signedValue(32, 2147483647)),
      signedValue(64, -INT64_C(4611686016279904256)));
  return passed;
}

bool testQ31RawHighProducts() {
  bool passed = true;
  passed &= expectEqual(
      "q31 max times max raw high",
      computeSignedRawHighProduct(signedValue(32, 2147483647), signedValue(32, 2147483647)),
      signedValue(32, 1073741823));
  passed &= expectEqual(
      "q31 min times min raw high",
      computeSignedRawHighProduct(signedValue(32, -2147483648LL), signedValue(32, -2147483648LL)),
      signedValue(32, 1073741824));
  passed &= expectEqual(
      "q31 min times max raw high",
      computeSignedRawHighProduct(signedValue(32, -2147483648LL), signedValue(32, 2147483647)),
      signedValue(32, -1073741824));
  passed &=
      expectEqual("q31 negative fraction raw high",
                  computeSignedRawHighProduct(signedValue(32, -2147483648LL), signedValue(32, 1)),
                  signedValue(32, -1));
  return passed;
}

bool testAccumulatorIntermediateWidth() {
  return getAccumulatorUpdateIntermediateWidth(40, 32) == 41 &&
         getAccumulatorUpdateIntermediateWidth(16, 32) == 33 &&
         getAccumulatorUpdateIntermediateWidth(64, 64) == 65;
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
    for (unsigned shift : {1U, 2U, accumulatorWidth}) {
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

int64_t updateReference(int64_t accumulator, int64_t product, unsigned width,
                        AccumulatorOverflowMode overflowMode) {
  int64_t updated = accumulator + product;
  if (overflowMode == AccumulatorOverflowMode::Wrap)
    return wrapSigned(updated, width);
  int64_t minimum = -(int64_t{1} << (width - 1));
  int64_t maximum = (int64_t{1} << (width - 1)) - 1;
  return std::clamp(updated, minimum, maximum);
}

int64_t exportReference(int64_t accumulator, unsigned shift, unsigned width, RoundingMode rounding,
                        AccumulatorOverflowMode overflowMode) {
  int64_t rounded = roundReference(accumulator, shift, rounding);
  if (overflowMode == AccumulatorOverflowMode::Wrap)
    return wrapSigned(rounded, width);
  int64_t minimum = -(int64_t{1} << (width - 1));
  int64_t maximum = (int64_t{1} << (width - 1)) - 1;
  return std::clamp(rounded, minimum, maximum);
}

bool testSignedUniformSosDf2Validation() {
  SignedUniformSosDf2Policy policy{3,
                                   2,
                                   6,
                                   AccumulatorOverflowMode::Saturate,
                                   RoundingMode::NearestEven,
                                   AccumulatorOverflowMode::Saturate,
                                   RoundingMode::TowardZero,
                                   AccumulatorOverflowMode::Wrap};
  llvm::APInt zero = signedValue(3, 0);
  auto evaluate = [&](const SignedUniformSosDf2Policy &candidate, const llvm::APInt &input) {
    return evaluateSignedUniformSosDf2Section(input, zero, zero, zero, zero, zero, zero, zero, zero,
                                              candidate);
  };
  if (!evaluate(policy, zero))
    return false;

  SignedUniformSosDf2Policy invalid = policy;
  invalid.storageWidth = 0;
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.fractionalBits = 4;
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.accumulatorWidth = 5;
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.updateOverflow = static_cast<AccumulatorOverflowMode>(2);
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.stateRounding = static_cast<RoundingMode>(3);
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.stateOverflow = static_cast<AccumulatorOverflowMode>(2);
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.outputRounding = static_cast<RoundingMode>(3);
  if (evaluate(invalid, zero))
    return false;
  invalid = policy;
  invalid.outputOverflow = static_cast<AccumulatorOverflowMode>(2);
  if (evaluate(invalid, zero))
    return false;
  return !evaluate(policy, signedValue(4, 0));
}

bool testSmallWidthSosDf2Exhaustive() {
  constexpr unsigned storageWidth = 3;
  constexpr unsigned fractionalBits = 2;
  constexpr unsigned accumulatorWidth = 6;
  constexpr int64_t minimum = -4;
  constexpr int64_t maximum = 3;
  constexpr int64_t scale = -3;
  constexpr int64_t b0 = 3;
  constexpr int64_t b1 = -2;
  constexpr int64_t b2 = 1;
  constexpr int64_t a1 = -1;
  constexpr int64_t a2 = 2;

  for (int64_t input = minimum; input <= maximum; ++input) {
    for (int64_t d1 = minimum; d1 <= maximum; ++d1) {
      for (int64_t d2 = minimum; d2 <= maximum; ++d2) {
        for (AccumulatorOverflowMode updateOverflow :
             {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
          for (RoundingMode stateRounding : {RoundingMode::TowardNegative,
                                             RoundingMode::NearestEven, RoundingMode::TowardZero}) {
            for (AccumulatorOverflowMode stateOverflow :
                 {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
              for (RoundingMode outputRounding :
                   {RoundingMode::TowardNegative, RoundingMode::NearestEven,
                    RoundingMode::TowardZero}) {
                for (AccumulatorOverflowMode outputOverflow :
                     {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
                  SignedUniformSosDf2Policy policy{storageWidth,   fractionalBits, accumulatorWidth,
                                                   updateOverflow, stateRounding,  stateOverflow,
                                                   outputRounding, outputOverflow};
                  auto actual = evaluateSignedUniformSosDf2Section(
                      signedValue(storageWidth, input), signedValue(storageWidth, scale),
                      signedValue(storageWidth, b0), signedValue(storageWidth, b1),
                      signedValue(storageWidth, b2), signedValue(storageWidth, a1),
                      signedValue(storageWidth, a2), signedValue(storageWidth, d1),
                      signedValue(storageWidth, d2), policy);
                  if (!actual)
                    return false;

                  int64_t stateAccumulator = 0;
                  stateAccumulator = updateReference(stateAccumulator, input * scale,
                                                     accumulatorWidth, updateOverflow);
                  stateAccumulator =
                      updateReference(stateAccumulator, d1 * a1, accumulatorWidth, updateOverflow);
                  stateAccumulator =
                      updateReference(stateAccumulator, d2 * a2, accumulatorWidth, updateOverflow);
                  int64_t nextD1 = exportReference(stateAccumulator, fractionalBits, storageWidth,
                                                   stateRounding, stateOverflow);
                  int64_t outputAccumulator = 0;
                  outputAccumulator = updateReference(outputAccumulator, nextD1 * b0,
                                                      accumulatorWidth, updateOverflow);
                  outputAccumulator =
                      updateReference(outputAccumulator, d1 * b1, accumulatorWidth, updateOverflow);
                  outputAccumulator =
                      updateReference(outputAccumulator, d2 * b2, accumulatorWidth, updateOverflow);
                  int64_t output = exportReference(outputAccumulator, fractionalBits, storageWidth,
                                                   outputRounding, outputOverflow);

                  if (!expectEqual("small-width SOS output", actual->output,
                                   signedValue(storageWidth, output)) ||
                      !expectEqual("small-width SOS d1", actual->d1,
                                   signedValue(storageWidth, nextD1)) ||
                      !expectEqual("small-width SOS d2", actual->d2, signedValue(storageWidth, d1)))
                    return false;
                }
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool testSosDf2ImpulseConvention() {
  constexpr unsigned storageWidth = 16;
  SignedUniformSosDf2Policy policy{storageWidth,
                                   0,
                                   32,
                                   AccumulatorOverflowMode::Saturate,
                                   RoundingMode::TowardNegative,
                                   AccumulatorOverflowMode::Saturate,
                                   RoundingMode::TowardNegative,
                                   AccumulatorOverflowMode::Saturate};
  const llvm::APInt zero = signedValue(storageWidth, 0);
  const llvm::APInt one = signedValue(storageWidth, 1);
  const llvm::APInt b0 = signedValue(storageWidth, 2);
  const llvm::APInt b1 = signedValue(storageWidth, 3);
  const llvm::APInt b2 = signedValue(storageWidth, 5);
  const llvm::APInt a1 = signedValue(storageWidth, 7);
  const llvm::APInt a2 = signedValue(storageWidth, 11);

  auto step0 = evaluateSignedUniformSosDf2Section(one, one, b0, b1, b2, a1, a2, zero, zero, policy);
  if (!step0 || !expectEqual("SOS impulse y0", step0->output, signedValue(storageWidth, 2)) ||
      !expectEqual("SOS impulse d1[0]", step0->d1, signedValue(storageWidth, 1)) ||
      !expectEqual("SOS impulse d2[0]", step0->d2, zero))
    return false;

  auto step1 = evaluateSignedUniformSosDf2Section(zero, one, b0, b1, b2, a1, a2, step0->d1,
                                                  step0->d2, policy);
  if (!step1 || !expectEqual("SOS impulse y1", step1->output, signedValue(storageWidth, 17)) ||
      !expectEqual("SOS impulse d1[1]", step1->d1, signedValue(storageWidth, 7)) ||
      !expectEqual("SOS impulse d2[1]", step1->d2, signedValue(storageWidth, 1)))
    return false;

  auto step2 = evaluateSignedUniformSosDf2Section(zero, one, b0, b1, b2, a1, a2, step1->d1,
                                                  step1->d2, policy);
  return step2 && expectEqual("SOS impulse y2", step2->output, signedValue(storageWidth, 146)) &&
         expectEqual("SOS impulse d1[2]", step2->d1, signedValue(storageWidth, 60)) &&
         expectEqual("SOS impulse d2[2]", step2->d2, signedValue(storageWidth, 7));
}

bool testQ15SosDf2Directed() {
  constexpr unsigned storageWidth = 16;
  constexpr unsigned fractionalBits = 15;
  constexpr unsigned accumulatorWidth = 40;
  constexpr int64_t input = -32768;
  constexpr int64_t scale = -32768;
  constexpr int64_t b0 = -32768;
  constexpr int64_t b1 = 32767;
  constexpr int64_t b2 = 32767;
  constexpr int64_t a1 = 32767;
  constexpr int64_t a2 = -32768;
  constexpr int64_t d1 = -32768;
  constexpr int64_t d2 = 32767;

  for (AccumulatorOverflowMode destinationOverflow :
       {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
    SignedUniformSosDf2Policy policy{
        storageWidth,
        fractionalBits,
        accumulatorWidth,
        AccumulatorOverflowMode::Saturate,
        RoundingMode::NearestEven,
        destinationOverflow,
        RoundingMode::TowardZero,
        destinationOverflow,
    };
    auto actual = evaluateSignedUniformSosDf2Section(
        signedValue(storageWidth, input), signedValue(storageWidth, scale),
        signedValue(storageWidth, b0), signedValue(storageWidth, b1), signedValue(storageWidth, b2),
        signedValue(storageWidth, a1), signedValue(storageWidth, a2), signedValue(storageWidth, d1),
        signedValue(storageWidth, d2), policy);
    if (!actual)
      return false;

    int64_t stateAccumulator = input * scale;
    stateAccumulator += d1 * a1;
    stateAccumulator += d2 * a2;
    int64_t nextD1 = exportReference(stateAccumulator, fractionalBits, storageWidth,
                                     policy.stateRounding, policy.stateOverflow);
    int64_t outputAccumulator = nextD1 * b0 + d1 * b1 + d2 * b2;
    int64_t output = exportReference(outputAccumulator, fractionalBits, storageWidth,
                                     policy.outputRounding, policy.outputOverflow);
    if (!expectEqual("Q15 SOS output", actual->output, signedValue(storageWidth, output)) ||
        !expectEqual("Q15 SOS d1", actual->d1, signedValue(storageWidth, nextD1)) ||
        !expectEqual("Q15 SOS d2", actual->d2, signedValue(storageWidth, d1)))
      return false;
  }
  return true;
}

bool testQ31SosDf2UpdateOverflow() {
  constexpr unsigned storageWidth = 32;
  constexpr unsigned fractionalBits = 31;
  constexpr unsigned accumulatorWidth = 64;
  constexpr int64_t minimumQ31 = INT64_C(-2147483648);
  const llvm::APInt zero = signedValue(storageWidth, 0);
  const llvm::APInt minimum = signedValue(storageWidth, minimumQ31);

  __int128 product = static_cast<__int128>(minimumQ31) * minimumQ31;
  __int128 exactUpdate = product + product;
  __int128 signedMinimum64 = -(static_cast<__int128>(1) << 63);
  __int128 signedMaximum64 = (static_cast<__int128>(1) << 63) - 1;
  if (exactUpdate != (static_cast<__int128>(1) << 63))
    return false;

  for (AccumulatorOverflowMode updateOverflow :
       {AccumulatorOverflowMode::Wrap, AccumulatorOverflowMode::Saturate}) {
    SignedUniformSosDf2Policy policy{storageWidth,
                                     fractionalBits,
                                     accumulatorWidth,
                                     updateOverflow,
                                     RoundingMode::TowardNegative,
                                     AccumulatorOverflowMode::Saturate,
                                     RoundingMode::TowardNegative,
                                     AccumulatorOverflowMode::Saturate};
    auto actual = evaluateSignedUniformSosDf2Section(minimum, minimum, zero, zero, zero, minimum,
                                                     zero, minimum, zero, policy);
    if (!actual)
      return false;

    __int128 updated = updateOverflow == AccumulatorOverflowMode::Wrap
                           ? exactUpdate - (static_cast<__int128>(1) << 64)
                           : std::clamp(exactUpdate, signedMinimum64, signedMaximum64);
    __int128 exported = updated / (static_cast<__int128>(1) << fractionalBits);
    exported =
        std::clamp(exported, static_cast<__int128>(INT32_MIN), static_cast<__int128>(INT32_MAX));
    if (!expectEqual("Q31 SOS 65-bit update", actual->d1,
                     signedValue(storageWidth, static_cast<int64_t>(exported))) ||
        !expectEqual("Q31 SOS zero output", actual->output, zero) ||
        !expectEqual("Q31 SOS state shift", actual->d2, minimum))
      return false;
  }
  return true;
}

} // namespace

int main() {
  bool passed = testQ15FullProducts();
  passed &= testCanonicalQ15TwiddleIdentities();
  passed &= testQ31FullProducts();
  passed &= testQ31RawHighProducts();
  passed &= testAccumulatorIntermediateWidth();
  passed &= testAccumulatorBoundaries();
  passed &= testAccumulatorChains();
  passed &= testAccumulatorExportRounding();
  passed &= testAccumulatorExportOverflow();
  passed &= testSmallWidthExhaustive();
  passed &= testSmallWidthExportExhaustive();
  passed &= testSignedUniformSosDf2Validation();
  passed &= testSmallWidthSosDf2Exhaustive();
  passed &= testSosDf2ImpulseConvention();
  passed &= testQ15SosDf2Directed();
  passed &= testQ31SosDf2UpdateOverflow();
  if (!passed)
    return 1;
  llvm::outs() << "fixed-point accumulator semantics: PASS\n";
  return 0;
}
