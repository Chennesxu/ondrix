#include "ondrix/Analysis/CanonicalTwiddleAnalysis.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Support/FixedPointSemantics.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <functional>

using ondrix::analysis::CanonicalPackedQ15TwiddleIdentity;
using ondrix::analysis::CanonicalPackedQ15TwiddleStatus;
using ondrix::analysis::classifyCanonicalPackedQ15Twiddle;
using ondrix::analysis::packedQ15ProductFitsNarrowCarrier;
using ondrix::analysis::planCanonicalPackedQ15Twiddle;
using ondrix::fixedpoint::AccumulatorOverflowMode;
using ondrix::fixedpoint::computeSignedFullProduct;
using ondrix::fixedpoint::exportSignedAccumulator;
using ondrix::fixedpoint::RoundingMode;

namespace {

struct Fixture {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::Block *block;
  ondrix::ondsp::CxLayoutAttr layout;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::ProductAttr product;
  ondrix::ondsp::ScaleAttr productScale;
  ondrix::ondsp::ScaleAttr outputScale;

  Fixture() : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))) {
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        ondrix::ondsp::OndspDialect>();
    mlir::OpBuilder builder(&context);
    mlir::Type i16 = builder.getI16Type();
    mlir::Type i32 = builder.getI32Type();
    layout = ondrix::ondsp::CxLayoutAttr::get(&context,
                                              ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo);
    numeric = ondrix::ondsp::FixedAttr::get(&context, ondrix::ondsp::Signedness::Signed, i16, 15);
    product = ondrix::ondsp::ProductAttr::get(&context, ondrix::ondsp::ProductSelection::Full);
    productScale =
        ondrix::ondsp::ScaleAttr::get(&context, 0, 15, ondrix::ondsp::RoundingMode::NearestEven,
                                      ondrix::ondsp::OverflowMode::Saturate, i16);
    outputScale =
        ondrix::ondsp::ScaleAttr::get(&context, 0, 1, ondrix::ondsp::RoundingMode::NearestEven,
                                      ondrix::ondsp::OverflowMode::Saturate, i16);

    builder.setInsertionPointToStart(module->getBody());
    function = builder.create<mlir::func::FuncOp>(module->getLoc(), "test",
                                                  builder.getFunctionType({i32, i32, i32}, {}));
    block = function.addEntryBlock();
  }

  mlir::Value constant(int32_t value, unsigned width = 32) {
    mlir::OpBuilder builder(block, block->end());
    return builder.create<mlir::arith::ConstantIntOp>(module->getLoc(), static_cast<int64_t>(value),
                                                      width);
  }

  ondrix::ondsp::CxButterflyOp create(mlir::Value twiddle) {
    mlir::OpBuilder builder(block, block->end());
    mlir::Type i32 = builder.getI32Type();
    return builder.create<ondrix::ondsp::CxButterflyOp>(
        module->getLoc(), mlir::TypeRange{i32, i32}, block->getArgument(0), block->getArgument(1),
        twiddle, layout, numeric, product, productScale, outputScale,
        ondrix::ondsp::CxButterflyVariantAttr());
  }
};

bool hasStatus(ondrix::ondsp::CxButterflyOp op, CanonicalPackedQ15TwiddleStatus status) {
  return classifyCanonicalPackedQ15Twiddle(op).status == status;
}

bool testClassificationReasons() {
  Fixture fixture;
  auto one = fixture.create(fixture.constant(32767));
  auto minusJ = fixture.create(fixture.constant(INT32_MIN));
  auto noncanonical = fixture.create(fixture.constant(0x40004000));
  auto runtime = fixture.create(fixture.block->getArgument(2));

  auto oneClassification = classifyCanonicalPackedQ15Twiddle(one);
  auto minusJClassification = classifyCanonicalPackedQ15Twiddle(minusJ);
  if (oneClassification.status != CanonicalPackedQ15TwiddleStatus::Authorized ||
      oneClassification.identity != CanonicalPackedQ15TwiddleIdentity::One ||
      minusJClassification.status != CanonicalPackedQ15TwiddleStatus::Authorized ||
      minusJClassification.identity != CanonicalPackedQ15TwiddleIdentity::MinusJ ||
      !hasStatus(noncanonical, CanonicalPackedQ15TwiddleStatus::NonCanonicalTwiddle) ||
      !hasStatus(runtime, CanonicalPackedQ15TwiddleStatus::NonConstantTwiddle))
    return false;

  one.setLayoutAttr(ondrix::ondsp::CxLayoutAttr::get(
      &fixture.context, ondrix::ondsp::ComplexLayout::PackedI16RealHiImagLo));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedLayout))
    return false;
  one.setLayoutAttr(fixture.layout);

  one.setNumericAttr(
      ondrix::ondsp::FixedAttr::get(&fixture.context, ondrix::ondsp::Signedness::Signed,
                                    mlir::IntegerType::get(&fixture.context, 32), 31));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedNumeric))
    return false;
  one.setNumericAttr(fixture.numeric);

  one.setProductAttr(
      ondrix::ondsp::ProductAttr::get(&fixture.context, ondrix::ondsp::ProductSelection::HighRaw));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedProduct))
    return false;
  one.setProductAttr(fixture.product);

  mlir::Type i16 = mlir::IntegerType::get(&fixture.context, 16);
  one.setProductScaleAttr(ondrix::ondsp::ScaleAttr::get(
      &fixture.context, 0, 14, ondrix::ondsp::RoundingMode::NearestEven,
      ondrix::ondsp::OverflowMode::Saturate, i16));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedScale))
    return false;
  one.setProductScaleAttr(fixture.productScale);

  one.setOutputScaleAttr(ondrix::ondsp::ScaleAttr::get(&fixture.context, 0, 1,
                                                       ondrix::ondsp::RoundingMode::TowardZero,
                                                       ondrix::ondsp::OverflowMode::Saturate, i16));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedRounding))
    return false;
  one.setOutputScaleAttr(fixture.outputScale);

  one.setOutputScaleAttr(ondrix::ondsp::ScaleAttr::get(&fixture.context, 0, 1,
                                                       ondrix::ondsp::RoundingMode::NearestEven,
                                                       ondrix::ondsp::OverflowMode::Wrap, i16));
  if (!hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedOverflow))
    return false;
  one.setOutputScaleAttr(fixture.outputScale);

  one.getTwiddleMutable().assign(fixture.constant(32767, 16));
  return hasStatus(one, CanonicalPackedQ15TwiddleStatus::UnsupportedValueDomain);
}

bool testSubjectBoundPlan() {
  Fixture fixture;
  auto first = fixture.create(fixture.constant(32767));
  auto second = fixture.create(fixture.constant(32767));

  auto wrongSubject = planCanonicalPackedQ15Twiddle(first);
  if (!wrongSubject ||
      mlir::succeeded(
          std::move(*wrongSubject).consumeIfValid(second, [](CanonicalPackedQ15TwiddleIdentity) {
            return mlir::success();
          })))
    return false;

  auto stale = planCanonicalPackedQ15Twiddle(first);
  if (!stale)
    return false;
  first.getTwiddleMutable().assign(fixture.constant(0));
  if (mlir::succeeded(std::move(*stale).consumeIfValid(
          first, [](CanonicalPackedQ15TwiddleIdentity) { return mlir::success(); })))
    return false;

  first.getTwiddleMutable().assign(fixture.constant(INT32_MIN));
  auto valid = planCanonicalPackedQ15Twiddle(first);
  bool consumed = false;
  if (!valid || mlir::failed(std::move(*valid).consumeIfValid(
                    first, [&](CanonicalPackedQ15TwiddleIdentity identity) {
                      consumed = identity == CanonicalPackedQ15TwiddleIdentity::MinusJ;
                      return mlir::success();
                    })))
    return false;
  return consumed && mlir::failed(std::move(*valid).consumeIfValid(
                         first, [](CanonicalPackedQ15TwiddleIdentity) { return mlir::success(); }));
}

bool testPolicyMutationInvalidation() {
  Fixture fixture;
  mlir::Type i16 = mlir::IntegerType::get(&fixture.context, 16);
  mlir::Type i32 = mlir::IntegerType::get(&fixture.context, 32);
  llvm::SmallVector<std::function<void(ondrix::ondsp::CxButterflyOp)>> mutations = {
      [&](ondrix::ondsp::CxButterflyOp op) {
        op.setLayoutAttr(ondrix::ondsp::CxLayoutAttr::get(
            &fixture.context, ondrix::ondsp::ComplexLayout::PackedI16RealHiImagLo));
      },
      [&](ondrix::ondsp::CxButterflyOp op) {
        op.setNumericAttr(ondrix::ondsp::FixedAttr::get(
            &fixture.context, ondrix::ondsp::Signedness::Signed, i32, 31));
      },
      [&](ondrix::ondsp::CxButterflyOp op) {
        op.setProductAttr(ondrix::ondsp::ProductAttr::get(
            &fixture.context, ondrix::ondsp::ProductSelection::HighRaw));
      },
      [&](ondrix::ondsp::CxButterflyOp op) {
        op.setProductScaleAttr(ondrix::ondsp::ScaleAttr::get(
            &fixture.context, 0, 14, ondrix::ondsp::RoundingMode::NearestEven,
            ondrix::ondsp::OverflowMode::Saturate, i16));
      },
      [&](ondrix::ondsp::CxButterflyOp op) {
        op.setOutputScaleAttr(ondrix::ondsp::ScaleAttr::get(
            &fixture.context, 0, 1, ondrix::ondsp::RoundingMode::TowardZero,
            ondrix::ondsp::OverflowMode::Saturate, i16));
      },
  };

  for (const auto &mutate : mutations) {
    auto butterfly = fixture.create(fixture.constant(32767));
    auto plan = planCanonicalPackedQ15Twiddle(butterfly);
    if (!plan)
      return false;
    mutate(butterfly);
    if (mlir::succeeded(std::move(*plan).consumeIfValid(
            butterfly, [](CanonicalPackedQ15TwiddleIdentity) { return mlir::success(); })))
      return false;
  }
  return true;
}

llvm::APInt signedValue(unsigned width, int64_t value) {
  return llvm::APInt(width, static_cast<uint64_t>(value), true);
}

llvm::APInt requantizeQ15(const llvm::APInt &value, const llvm::APInt &twiddle) {
  return exportSignedAccumulator(computeSignedFullProduct(value, twiddle), 15, 16,
                                 RoundingMode::NearestEven, AccumulatorOverflowMode::Saturate);
}

llvm::APInt requantizeNegatedQ15Product(const llvm::APInt &value, const llvm::APInt &twiddle) {
  llvm::APInt product = computeSignedFullProduct(value, twiddle).sext(33);
  return exportSignedAccumulator(-product, 15, 16, RoundingMode::NearestEven,
                                 AccumulatorOverflowMode::Saturate);
}

bool testExhaustiveGroundTruth() {
  Fixture fixture;
  auto one = fixture.create(fixture.constant(32767));
  auto minusJ = fixture.create(fixture.constant(INT32_MIN));
  auto onePlan = planCanonicalPackedQ15Twiddle(one);
  auto minusJPlan = planCanonicalPackedQ15Twiddle(minusJ);
  if (!onePlan || !minusJPlan)
    return false;

  bool oneAuthorized = false;
  bool minusJAuthorized = false;
  if (mlir::failed(std::move(*onePlan).consumeIfValid(
          one,
          [&](CanonicalPackedQ15TwiddleIdentity identity) {
            oneAuthorized = identity == CanonicalPackedQ15TwiddleIdentity::One;
            return mlir::success();
          })) ||
      mlir::failed(std::move(*minusJPlan)
                       .consumeIfValid(minusJ,
                                       [&](CanonicalPackedQ15TwiddleIdentity identity) {
                                         minusJAuthorized =
                                             identity == CanonicalPackedQ15TwiddleIdentity::MinusJ;
                                         return mlir::success();
                                       })) ||
      !oneAuthorized || !minusJAuthorized)
    return false;

  const llvm::APInt oneRaw = signedValue(16, 32767);
  const llvm::APInt minusOneRaw = signedValue(16, -32768);
  for (int64_t value = INT16_MIN; value <= INT16_MAX; ++value) {
    llvm::APInt input = signedValue(16, value);
    int64_t specializedOne = value + (value < -16384 ? 1 : 0) - (value > 16384 ? 1 : 0);
    if (requantizeQ15(input, oneRaw) != signedValue(16, specializedOne))
      return false;

    int64_t specializedNegate = value == INT16_MIN ? INT16_MAX : -value;
    if (requantizeQ15(input, minusOneRaw) != signedValue(16, specializedNegate))
      return false;
    if (requantizeNegatedQ15Product(input, minusOneRaw) != input)
      return false;
  }
  return true;
}

// The narrow-carrier query's whole content is one inequality, so the bound is
// pinned here rather than inferred from a codegen witness: the cross sums are
// bounded by 32768 * (|wr| + |wi|), and 65535 is the largest sum an i32
// carrier holds.
bool testNarrowCarrierBound() {
  constexpr int64_t kInt32Max = 2147483647;
  constexpr int64_t kComponentMagnitude = 32768;
  // The arithmetic claim, swept rather than asserted: at the admitted bound
  // every cross sum over the extreme component pairs fits, and one unit past
  // it the imaginary sum is exactly 2^31.
  for (int64_t real = -32768; real <= 32767; ++real) {
    int64_t remaining = 65535 - (real < 0 ? -real : real);
    if (remaining < 0 || remaining > 32767)
      continue;
    for (int64_t imaginary : {remaining, -remaining}) {
      for (int64_t br : {int64_t{-32768}, int64_t{32767}}) {
        for (int64_t bi : {int64_t{-32768}, int64_t{32767}}) {
          if (br * real - bi * imaginary > kInt32Max || br * imaginary + bi * real > kInt32Max ||
              br * real - bi * imaginary < -kInt32Max - 1 ||
              br * imaginary + bi * real < -kInt32Max - 1)
            return false;
        }
      }
    }
  }
  if (-kComponentMagnitude * -kComponentMagnitude + -kComponentMagnitude * -kComponentMagnitude !=
      kInt32Max + 1)
    return false;

  // And the query itself, at the boundary and at the two failing shapes.
  Fixture fixture;
  auto packed = [](int32_t real, int32_t imaginary) {
    return static_cast<int32_t>((static_cast<uint32_t>(imaginary) << 16) |
                                (static_cast<uint32_t>(real) & 0xFFFFU));
  };
  // The 45-degree twiddle: what every quantized unit-modulus value looks like.
  if (!packedQ15ProductFitsNarrowCarrier(fixture.create(fixture.constant(packed(23170, -23170)))))
    return false;
  // Exactly at the bound, 32767 + 32768.
  if (!packedQ15ProductFitsNarrowCarrier(fixture.create(fixture.constant(packed(32767, -32768)))))
    return false;
  // One unit past it: the only pair a 32-bit carrier cannot hold.
  if (packedQ15ProductFitsNarrowCarrier(fixture.create(fixture.constant(packed(-32768, -32768)))))
    return false;
  // A runtime twiddle carries no value, so it fails closed.
  if (packedQ15ProductFitsNarrowCarrier(fixture.create(fixture.block->getArgument(2))))
    return false;
  return true;
}

} // namespace

int main() {
  if (!testClassificationReasons() || !testSubjectBoundPlan() ||
      !testPolicyMutationInvalidation() || !testExhaustiveGroundTruth() ||
      !testNarrowCarrierBound()) {
    llvm::errs() << "canonical twiddle analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "canonical twiddle analysis: PASS\n";
  return 0;
}
