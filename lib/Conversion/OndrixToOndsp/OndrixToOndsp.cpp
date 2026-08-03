#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/DctCoefficients.h"
#include "ondrix/Support/FirStreamRuntimeShape.h"
#include "ondrix/Support/GuardedQ15Quantization.h"
#include "ondrix/Support/Q31TwiddleTables.h"

#include "llvm/ADT/APInt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDRIXTOONDSP
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static Value createReductionZero(Location loc, Type resultType,
                                 ConversionPatternRewriter &rewriter) {
  if (isa<ondrix::ondsp::AccType>(resultType))
    return rewriter.create<ondrix::ondsp::AccZeroOp>(loc, resultType);
  return rewriter.create<arith::ConstantOp>(loc, resultType, rewriter.getZeroAttr(resultType));
}

static Value createScalarFpDot(Location loc, Value lhs, Value rhs, ondrix::ondsp::FpAttr numeric,
                               ConversionPatternRewriter &rewriter) {
  switch (numeric.getContract()) {
  case ondrix::ondsp::FpContractMode::Off:
    return rewriter.create<arith::MulFOp>(loc, lhs, rhs);
  case ondrix::ondsp::FpContractMode::Fma: {
    Value zero = rewriter.create<arith::ConstantOp>(loc, numeric.getFormat(),
                                                    rewriter.getZeroAttr(numeric.getFormat()));
    return rewriter.create<math::FmaOp>(loc, lhs, rhs, zero);
  }
  case ondrix::ondsp::FpContractMode::Fast: {
    Value zero = rewriter.create<arith::ConstantOp>(loc, numeric.getFormat(),
                                                    rewriter.getZeroAttr(numeric.getFormat()));
    return rewriter.create<math::FmaOp>(loc, lhs, rhs, zero, ondrix::ondsp::getFastContractFlags());
  }
  }
  llvm_unreachable("unknown floating-point contract mode");
}

static Value createFpAccumulatorUpdate(Location loc, Value lhs, Value rhs, Value accumulator,
                                       ondrix::ondsp::FpAttr numeric, OpBuilder &builder) {
  switch (numeric.getContract()) {
  case ondrix::ondsp::FpContractMode::Off: {
    Value product = builder.create<arith::MulFOp>(loc, lhs, rhs);
    return builder.create<arith::AddFOp>(loc, accumulator, product);
  }
  case ondrix::ondsp::FpContractMode::Fma:
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator);
  case ondrix::ondsp::FpContractMode::Fast:
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator,
                                       ondrix::ondsp::getFastContractFlags());
  }
  llvm_unreachable("unknown floating-point contract mode");
}

static Value createFpMultiply(Location loc, Value lhs, Value rhs, ondrix::ondsp::FpAttr numeric,
                              OpBuilder &builder) {
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return builder.create<arith::MulFOp>(loc, lhs, rhs, ondrix::ondsp::getFastContractFlags());
  return builder.create<arith::MulFOp>(loc, lhs, rhs);
}

static Value createFpAdd(Location loc, Value lhs, Value rhs, ondrix::ondsp::FpAttr numeric,
                         OpBuilder &builder) {
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return builder.create<arith::AddFOp>(loc, lhs, rhs, ondrix::ondsp::getFastContractFlags());
  return builder.create<arith::AddFOp>(loc, lhs, rhs);
}

static void assertValidFirFilterShape(Location loc, Value inputLength, Value coefficientLength,
                                      Value outputLength, Value zero, Value one,
                                      OpBuilder &builder) {
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients, builder.getStringAttr("valid FIR requires at least one coefficient"));

  Value inputCoversWindow =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, coefficientLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversWindow,
      builder.getStringAttr("valid FIR input must cover one coefficient window"));

  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, coefficientLength);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, remaining, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "valid FIR output length must equal input length minus coefficient length plus one"));
}

static void assertValidConv1DShape(Location loc, Value inputLength, Value kernelLength,
                                   Value outputLength, Value zero, Value one, OpBuilder &builder) {
  Value hasKernel =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, kernelLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasKernel, builder.getStringAttr("conv1d requires at least one kernel element"));
  Value inputCoversKernel =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, kernelLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversKernel,
      builder.getStringAttr("conv1d input must cover one complete kernel window"));
  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, kernelLength);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, remaining, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "conv1d output length must equal input length minus kernel length plus one"));
}

static void assertValidFirDecimateShape(Location loc, Value inputLength, Value coefficientLength,
                                        Value outputLength, Value factor, Value zero, Value one,
                                        OpBuilder &builder) {
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients,
      builder.getStringAttr("fir_decimate requires at least one coefficient"));
  Value inputCoversCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, inputLength, coefficientLength);
  builder.create<cf::AssertOp>(
      loc, inputCoversCoefficients,
      builder.getStringAttr("fir_decimate input must cover one complete coefficient window"));
  Value remaining = builder.create<arith::SubIOp>(loc, inputLength, coefficientLength);
  Value completeSteps = builder.create<arith::DivUIOp>(loc, remaining, factor);
  Value requiredOutputLength = builder.create<arith::AddIOp>(loc, completeSteps, one);
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "fir_decimate output length must equal floor((input length - coefficient length) / "
          "factor) plus one"));
}

static void assertFirInterpolateShape(Location loc, Value inputLength, Value coefficientLength,
                                      Value outputLength, Value factor, Value zero, Value one,
                                      OpBuilder &builder) {
  Value hasInput = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasInput, builder.getStringAttr("fir_interpolate requires at least one input sample"));
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients,
      builder.getStringAttr("fir_interpolate requires at least one coefficient"));

  Value inputIntervals = builder.create<arith::SubIOp>(loc, inputLength, one);
  Value scaledIntervals = builder.create<arith::MulIOp>(loc, inputIntervals, factor);
  Value recoveredIntervals = builder.create<arith::DivUIOp>(loc, scaledIntervals, factor);
  Value multiplicationFits = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                           recoveredIntervals, inputIntervals);
  builder.create<cf::AssertOp>(
      loc, multiplicationFits,
      builder.getStringAttr("fir_interpolate result length multiplication must not overflow"));

  Value requiredOutputLength =
      builder.create<arith::AddIOp>(loc, scaledIntervals, coefficientLength);
  Value additionFits = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                     requiredOutputLength, scaledIntervals);
  builder.create<cf::AssertOp>(
      loc, additionFits,
      builder.getStringAttr("fir_interpolate result length addition must not overflow"));
  Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outputLength,
                                                      requiredOutputLength);
  builder.create<cf::AssertOp>(
      loc, outputMatches,
      builder.getStringAttr(
          "fir_interpolate output length must equal (input length - 1) * factor plus "
          "coefficient length"));
}

static void assertFullFirFilterShape(Location loc, Value inputLength, Value coefficientLength,
                                     Value outputLength, Value zero, Value one,
                                     OpBuilder &builder) {
  Value hasInput = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasInput, builder.getStringAttr("full FIR requires at least one input sample"));
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(loc, hasCoefficients,
                               builder.getStringAttr("full FIR requires at least one coefficient"));

  Value leftPadding = builder.create<arith::SubIOp>(loc, coefficientLength, one);
  Value outputCoversPadding =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, outputLength, leftPadding);
  Value recoveredInput = builder.create<arith::SubIOp>(loc, outputLength, leftPadding);
  Value outputMatches =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, recoveredInput, inputLength);
  Value validOutputShape = builder.create<arith::AndIOp>(loc, outputCoversPadding, outputMatches);
  builder.create<cf::AssertOp>(
      loc, validOutputShape,
      builder.getStringAttr(
          "full FIR output length must equal input length plus coefficient length minus one"));
}

static void assertFullFirFilterTileShape(Location loc, Value inputLength, Value coefficientLength,
                                         Value outputLength, Value outputOrigin, Value zero,
                                         Value one, OpBuilder &builder) {
  Value hasInput = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasInput, builder.getStringAttr("full FIR requires at least one input sample"));
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(loc, hasCoefficients,
                               builder.getStringAttr("full FIR requires at least one coefficient"));

  Value leftPadding = builder.create<arith::SubIOp>(loc, coefficientLength, one);
  Value completeOutputLength = builder.create<arith::AddIOp>(loc, inputLength, leftPadding);
  Value extentDidNotOverflow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                             completeOutputLength, inputLength);
  Value originInRange = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, outputOrigin,
                                                      completeOutputLength);
  Value remaining = builder.create<arith::SubIOp>(loc, completeOutputLength, outputOrigin);
  Value tileInRange =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, outputLength, remaining);
  Value validRange = builder.create<arith::AndIOp>(loc, extentDidNotOverflow, originInRange);
  validRange = builder.create<arith::AndIOp>(loc, validRange, tileInRange);
  builder.create<cf::AssertOp>(
      loc, validRange,
      builder.getStringAttr("full FIR output tile must lie within the complete output range"));
}

static Value createEmptyTensor(Location loc, RankedTensorType type, Value dynamicLength,
                               OpBuilder &builder) {
  SmallVector<Value> dynamicSizes;
  if (type.isDynamicDim(0))
    dynamicSizes.push_back(dynamicLength);
  return builder.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(), dynamicSizes);
}

static void assertValidSosSectionShape(Location loc, Value coefficientSections, Value scaleSections,
                                       Value stateSections, Value zero, OpBuilder &builder) {
  Value hasSections =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientSections, zero);
  builder.create<cf::AssertOp>(loc, hasSections,
                               builder.getStringAttr("SOS filter requires at least one section"));
  Value scalesMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                    coefficientSections, scaleSections);
  Value stateMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                     coefficientSections, stateSections);
  Value sectionsMatch = builder.create<arith::AndIOp>(loc, scalesMatch, stateMatches);
  builder.create<cf::AssertOp>(
      loc, sectionsMatch,
      builder.getStringAttr("SOS coefficient, scale, and state section counts must match"));
}

static Value createFirStreamInitialAccumulator(ondrix::ir::FirStreamOp op, Location loc,
                                               OpBuilder &builder) {
  if (isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return builder.create<ondrix::ondsp::AccZeroOp>(loc, *op.getAccumulator());
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  return builder.create<arith::ConstantOp>(loc, fp.getFormat(),
                                           builder.getZeroAttr(fp.getFormat()));
}

static Value exportFirStreamSample(ondrix::ir::FirStreamOp op, Value accumulator, Location loc,
                                   OpBuilder &builder) {
  if (!isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return accumulator;
  return builder.create<ondrix::ondsp::AccExportOp>(loc, op.getDst()->getStorage(), accumulator,
                                                    *op.getDst(), *op.getRounding(),
                                                    *op.getOverflow());
}

class FirOpLowering final : public OpConversionPattern<ondrix::ir::FirOp> {
public:
  using OpConversionPattern<ondrix::ir::FirOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value initial = createReductionZero(op.getLoc(), op.getResult().getType(), rewriter);
    auto replacement = rewriter.create<ondrix::ondsp::ReduceMacOp>(
        op.getLoc(), op.getResult().getType(), initial, adaptor.getInput(), adaptor.getCoeffs(),
        op.getNumeric(), op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class FirFilterOpLowering final : public OpConversionPattern<ondrix::ir::FirFilterOp> {
public:
  using OpConversionPattern<ondrix::ir::FirFilterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirFilterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientLength = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value outputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInit(), zero);
    Value outputOrigin = adaptor.getOutputOrigin();
    if (op.getBoundary() == ondrix::ir::FirBoundaryMode::Valid)
      assertValidFirFilterShape(loc, inputLength, coefficientLength, outputLength, zero, one,
                                rewriter);
    else if (outputOrigin)
      assertFullFirFilterTileShape(loc, inputLength, coefficientLength, outputLength, outputOrigin,
                                   zero, one, rewriter);
    else
      assertFullFirFilterShape(loc, inputLength, coefficientLength, outputLength, zero, one,
                               rewriter);

    Value leftPadding;
    if (op.getBoundary() == ondrix::ir::FirBoundaryMode::Full)
      leftPadding = rewriter.create<arith::SubIOp>(loc, coefficientLength, one);

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange outputArgs) {
          Value globalOutputIndex = outputIndex;
          if (outputOrigin)
            globalOutputIndex = builder.create<arith::AddIOp>(bodyLoc, outputOrigin, outputIndex);
          Value initial;
          if (fixed)
            initial = builder.create<ondrix::ondsp::AccZeroOp>(bodyLoc, *op.getAccumulator());
          else
            initial = builder.create<arith::ConstantOp>(bodyLoc, fp.getFormat(),
                                                        builder.getZeroAttr(fp.getFormat()));

          Value firstValidTap;
          Value inputBase;
          if (op.getBoundary() == ondrix::ir::FirBoundaryMode::Full) {
            Value outputBeforeLeft = builder.create<arith::CmpIOp>(
                bodyLoc, arith::CmpIPredicate::ult, globalOutputIndex, leftPadding);
            Value leftDeficit =
                builder.create<arith::SubIOp>(bodyLoc, leftPadding, globalOutputIndex);
            firstValidTap =
                builder.create<arith::SelectOp>(bodyLoc, outputBeforeLeft, leftDeficit, zero);
            inputBase = builder.create<arith::SubIOp>(bodyLoc, globalOutputIndex, leftPadding);
          }

          auto tapLoop = builder.create<scf::ForOp>(
              bodyLoc, zero, coefficientLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange accumulatorArgs) {
                Value next;
                if (op.getBoundary() == ondrix::ir::FirBoundaryMode::Valid) {
                  Value inputIndex =
                      tapBuilder.create<arith::AddIOp>(tapLoc, globalOutputIndex, tap);
                  Value inputValue = tapBuilder.create<tensor::ExtractOp>(
                      tapLoc, adaptor.getInput(), ValueRange{inputIndex});
                  Value coefficient = tapBuilder.create<tensor::ExtractOp>(
                      tapLoc, adaptor.getCoeffs(), ValueRange{tap});
                  if (fixed) {
                    next = tapBuilder.create<ondrix::ondsp::MacOp>(
                        tapLoc, *op.getAccumulator(), accumulatorArgs.front(), inputValue,
                        coefficient, fixed, *op.getProduct());
                  } else {
                    next = createFpAccumulatorUpdate(tapLoc, inputValue, coefficient,
                                                     accumulatorArgs.front(), fp, tapBuilder);
                  }
                } else {
                  Value pastLeftPadding = tapBuilder.create<arith::CmpIOp>(
                      tapLoc, arith::CmpIPredicate::uge, tap, firstValidTap);
                  Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, inputBase, tap);
                  Value beforeRightPadding = tapBuilder.create<arith::CmpIOp>(
                      tapLoc, arith::CmpIPredicate::ult, inputIndex, inputLength);
                  Value inBounds =
                      tapBuilder.create<arith::AndIOp>(tapLoc, pastLeftPadding, beforeRightPadding);
                  auto guarded = tapBuilder.create<scf::IfOp>(
                      tapLoc, TypeRange{accumulatorArgs.front().getType()}, inBounds,
                      /*withElseRegion=*/true);
                  OpBuilder thenBuilder = guarded.getThenBodyBuilder();
                  Value inputValue = thenBuilder.create<tensor::ExtractOp>(
                      tapLoc, adaptor.getInput(), ValueRange{inputIndex});
                  Value coefficient = thenBuilder.create<tensor::ExtractOp>(
                      tapLoc, adaptor.getCoeffs(), ValueRange{tap});
                  Value updated;
                  if (fixed) {
                    updated = thenBuilder.create<ondrix::ondsp::MacOp>(
                        tapLoc, *op.getAccumulator(), accumulatorArgs.front(), inputValue,
                        coefficient, fixed, *op.getProduct());
                  } else {
                    updated = createFpAccumulatorUpdate(tapLoc, inputValue, coefficient,
                                                        accumulatorArgs.front(), fp, thenBuilder);
                  }
                  thenBuilder.create<scf::YieldOp>(tapLoc, updated);
                  OpBuilder elseBuilder = guarded.getElseBodyBuilder();
                  elseBuilder.create<scf::YieldOp>(tapLoc, accumulatorArgs.front());
                  next = guarded.getResult(0);
                }
                tapBuilder.create<scf::YieldOp>(tapLoc, next);
              });

          Value sample = tapLoop.getResult(0);
          if (fixed) {
            sample = builder.create<ondrix::ondsp::AccExportOp>(
                bodyLoc, op.getDst()->getStorage(), sample, *op.getDst(), *op.getRounding(),
                *op.getOverflow());
          }
          Value updated = builder.create<tensor::InsertOp>(bodyLoc, sample, outputArgs.front(),
                                                           ValueRange{outputIndex});
          builder.create<scf::YieldOp>(bodyLoc, updated);
        });

    rewriter.replaceOp(op, outputLoop.getResult(0));
    return success();
  }
};

template <typename OpTy>
static Value createResamplingSeed(OpTy op, ondrix::ondsp::FpAttr fp, OpBuilder &builder,
                                  Location loc) {
  if (!fp)
    return builder.create<ondrix::ondsp::AccZeroOp>(loc, *op.getAccumulator());
  return builder.create<arith::ConstantOp>(loc, fp.getFormat(),
                                           builder.getZeroAttr(fp.getFormat()));
}

template <typename OpTy>
static Value updateResamplingAccumulator(OpTy op, ondrix::ondsp::FixedAttr fixed,
                                         ondrix::ondsp::FpAttr fp, Value accumulator, Value input,
                                         Value coefficient, OpBuilder &builder, Location loc) {
  if (fp)
    return createFpAccumulatorUpdate(loc, input, coefficient, accumulator, fp, builder);
  return builder.create<ondrix::ondsp::MacOp>(loc, *op.getAccumulator(), accumulator, input,
                                              coefficient, fixed, *op.getProduct());
}

template <typename OpTy>
static Value exportResamplingSample(OpTy op, ondrix::ondsp::FpAttr fp, Value accumulator,
                                    OpBuilder &builder, Location loc) {
  if (fp)
    return accumulator;
  return builder.create<ondrix::ondsp::AccExportOp>(loc, op.getDst()->getStorage(), accumulator,
                                                    *op.getDst(), *op.getRounding(),
                                                    *op.getOverflow());
}

class FirDecimateOpLowering final : public OpConversionPattern<ondrix::ir::FirDecimateOp> {
public:
  using OpConversionPattern<ondrix::ir::FirDecimateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirDecimateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    int64_t factorValue = op.getFactorAttr().getValue().getSExtValue();
    Value factor = rewriter.create<arith::ConstantIndexOp>(loc, factorValue);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientLength = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value outputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInit(), zero);
    assertValidFirDecimateShape(loc, inputLength, coefficientLength, outputLength, factor, zero,
                                one, rewriter);

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location outputLoc, Value outputIndex, ValueRange outputArgs) {
          Value inputOrigin = builder.create<arith::MulIOp>(outputLoc, outputIndex, factor);
          Value initial = createResamplingSeed(op, fp, builder, outputLoc);
          auto tapLoop = builder.create<scf::ForOp>(
              outputLoc, zero, coefficientLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange tapArgs) {
                Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, inputOrigin, tap);
                Value input =
                    tapBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getInput(), inputIndex);
                Value coefficient =
                    tapBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getCoeffs(), tap);
                Value updated = updateResamplingAccumulator(op, fixed, fp, tapArgs.front(), input,
                                                            coefficient, tapBuilder, tapLoc);
                tapBuilder.create<scf::YieldOp>(tapLoc, updated);
              });
          Value output = exportResamplingSample(op, fp, tapLoop.getResult(0), builder, outputLoc);
          Value next =
              builder.create<tensor::InsertOp>(outputLoc, output, outputArgs.front(), outputIndex);
          builder.create<scf::YieldOp>(outputLoc, next);
        });
    rewriter.replaceOp(op, outputLoop.getResult(0));
    return success();
  }
};

class FirInterpolateOpLowering final : public OpConversionPattern<ondrix::ir::FirInterpolateOp> {
public:
  using OpConversionPattern<ondrix::ir::FirInterpolateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirInterpolateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    int64_t factorValue = op.getFactorAttr().getValue().getSExtValue();
    Value factor = rewriter.create<arith::ConstantIndexOp>(loc, factorValue);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientLength = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value outputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInit(), zero);
    assertFirInterpolateShape(loc, inputLength, coefficientLength, outputLength, factor, zero, one,
                              rewriter);

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location outputLoc, Value outputIndex, ValueRange outputArgs) {
          Value initial = createResamplingSeed(op, fp, builder, outputLoc);
          auto tapLoop = builder.create<scf::ForOp>(
              outputLoc, zero, coefficientLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange tapArgs) {
                Value outputCoversTap = tapBuilder.create<arith::CmpIOp>(
                    tapLoc, arith::CmpIPredicate::uge, outputIndex, tap);
                Value upsampledIndex = tapBuilder.create<arith::SubIOp>(tapLoc, outputIndex, tap);
                Value phase = tapBuilder.create<arith::RemUIOp>(tapLoc, upsampledIndex, factor);
                Value isInputPhase =
                    tapBuilder.create<arith::CmpIOp>(tapLoc, arith::CmpIPredicate::eq, phase, zero);
                Value inputIndex =
                    tapBuilder.create<arith::DivUIOp>(tapLoc, upsampledIndex, factor);
                Value beforeInputEnd = tapBuilder.create<arith::CmpIOp>(
                    tapLoc, arith::CmpIPredicate::ult, inputIndex, inputLength);
                Value inPhaseAndBounds =
                    tapBuilder.create<arith::AndIOp>(tapLoc, isInputPhase, beforeInputEnd);
                Value contributes =
                    tapBuilder.create<arith::AndIOp>(tapLoc, outputCoversTap, inPhaseAndBounds);
                auto guarded = tapBuilder.create<scf::IfOp>(
                    tapLoc, TypeRange{tapArgs.front().getType()}, contributes,
                    /*withElseRegion=*/true);
                OpBuilder thenBuilder = guarded.getThenBodyBuilder();
                Value input =
                    thenBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getInput(), inputIndex);
                Value coefficient =
                    thenBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getCoeffs(), tap);
                Value updated = updateResamplingAccumulator(op, fixed, fp, tapArgs.front(), input,
                                                            coefficient, thenBuilder, tapLoc);
                thenBuilder.create<scf::YieldOp>(tapLoc, updated);
                OpBuilder elseBuilder = guarded.getElseBodyBuilder();
                elseBuilder.create<scf::YieldOp>(tapLoc, tapArgs.front());
                tapBuilder.create<scf::YieldOp>(tapLoc, guarded.getResult(0));
              });
          Value output = exportResamplingSample(op, fp, tapLoop.getResult(0), builder, outputLoc);
          Value next =
              builder.create<tensor::InsertOp>(outputLoc, output, outputArgs.front(), outputIndex);
          builder.create<scf::YieldOp>(outputLoc, next);
        });
    rewriter.replaceOp(op, outputLoop.getResult(0));
    return success();
  }
};

class Conv1DOpLowering final : public OpConversionPattern<ondrix::ir::Conv1DOp> {
public:
  using OpConversionPattern<ondrix::ir::Conv1DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::Conv1DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value kernelLength = rewriter.create<tensor::DimOp>(loc, adaptor.getKernel(), zero);
    Value outputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInit(), zero);
    assertValidConv1DShape(loc, inputLength, kernelLength, outputLength, zero, one, rewriter);

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange outputArgs) {
          Value initial;
          if (fixed)
            initial = builder.create<ondrix::ondsp::AccZeroOp>(bodyLoc, *op.getAccumulator());
          else
            initial = builder.create<arith::ConstantOp>(bodyLoc, fp.getFormat(),
                                                        builder.getZeroAttr(fp.getFormat()));

          auto tapLoop = builder.create<scf::ForOp>(
              bodyLoc, zero, kernelLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange accumulatorArgs) {
                Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, outputIndex, tap);
                Value kernelIndex = tap;
                if (op.getMode() == ondrix::ir::Conv1DMode::Convolution) {
                  Value lastKernelIndex =
                      tapBuilder.create<arith::SubIOp>(tapLoc, kernelLength, one);
                  kernelIndex = tapBuilder.create<arith::SubIOp>(tapLoc, lastKernelIndex, tap);
                }
                Value inputValue = tapBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getInput(),
                                                                        ValueRange{inputIndex});
                Value kernelValue = tapBuilder.create<tensor::ExtractOp>(
                    tapLoc, adaptor.getKernel(), ValueRange{kernelIndex});
                Value next;
                if (fixed) {
                  next = tapBuilder.create<ondrix::ondsp::MacOp>(
                      tapLoc, *op.getAccumulator(), accumulatorArgs.front(), inputValue,
                      kernelValue, fixed, *op.getProduct());
                } else {
                  next = createFpAccumulatorUpdate(tapLoc, inputValue, kernelValue,
                                                   accumulatorArgs.front(), fp, tapBuilder);
                }
                tapBuilder.create<scf::YieldOp>(tapLoc, next);
              });

          Value sample = tapLoop.getResult(0);
          if (fixed) {
            sample = builder.create<ondrix::ondsp::AccExportOp>(
                bodyLoc, op.getDst()->getStorage(), sample, *op.getDst(), *op.getRounding(),
                *op.getOverflow());
          }
          Value updated = builder.create<tensor::InsertOp>(bodyLoc, sample, outputArgs.front(),
                                                           ValueRange{outputIndex});
          builder.create<scf::YieldOp>(bodyLoc, updated);
        });

    rewriter.replaceOp(op, outputLoop.getResult(0));
    return success();
  }
};

class FirStreamOpLowering final : public OpConversionPattern<ondrix::ir::FirStreamOp> {
public:
  using OpConversionPattern<ondrix::ir::FirStreamOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirStreamOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientLength = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value stateLength = rewriter.create<tensor::DimOp>(loc, adaptor.getState(), zero);

    ondrix::emitFirStreamRuntimeShapeAssertions(op, inputLength, coefficientLength, stateLength,
                                                zero, one, rewriter);

    Value emptyOutput = createEmptyTensor(loc, op.getOutput().getType(), inputLength, rewriter);
    Value emptyNextState =
        createEmptyTensor(loc, op.getNextState().getType(), stateLength, rewriter);
    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());

    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, inputLength, one, ValueRange{emptyOutput},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange outputArgs) {
          Value initial = createFirStreamInitialAccumulator(op, bodyLoc, builder);
          auto tapLoop = builder.create<scf::ForOp>(
              bodyLoc, zero, coefficientLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange accumulatorArgs) {
                Value extendedIndex = tapBuilder.create<arith::AddIOp>(tapLoc, outputIndex, tap);
                Value fromState = tapBuilder.create<arith::CmpIOp>(
                    tapLoc, arith::CmpIPredicate::ult, extendedIndex, stateLength);
                auto selected = tapBuilder.create<scf::IfOp>(
                    tapLoc, TypeRange{op.getInput().getType().getElementType()}, fromState,
                    /*withElseRegion=*/true);
                OpBuilder stateBuilder = selected.getThenBodyBuilder();
                Value stateValue = stateBuilder.create<tensor::ExtractOp>(
                    tapLoc, adaptor.getState(), ValueRange{extendedIndex});
                stateBuilder.create<scf::YieldOp>(tapLoc, stateValue);
                OpBuilder inputBuilder = selected.getElseBodyBuilder();
                Value inputIndex =
                    inputBuilder.create<arith::SubIOp>(tapLoc, extendedIndex, stateLength);
                Value inputValue = inputBuilder.create<tensor::ExtractOp>(
                    tapLoc, adaptor.getInput(), ValueRange{inputIndex});
                inputBuilder.create<scf::YieldOp>(tapLoc, inputValue);

                Value coefficient = tapBuilder.create<tensor::ExtractOp>(
                    tapLoc, adaptor.getCoeffs(), ValueRange{tap});
                Value next;
                if (fixed) {
                  next = tapBuilder.create<ondrix::ondsp::MacOp>(
                      tapLoc, *op.getAccumulator(), accumulatorArgs.front(), selected.getResult(0),
                      coefficient, fixed, *op.getProduct());
                } else {
                  next = createFpAccumulatorUpdate(tapLoc, selected.getResult(0), coefficient,
                                                   accumulatorArgs.front(), fp, tapBuilder);
                }
                tapBuilder.create<scf::YieldOp>(tapLoc, next);
              });
          Value sample = exportFirStreamSample(op, tapLoop.getResult(0), bodyLoc, builder);
          Value updated = builder.create<tensor::InsertOp>(bodyLoc, sample, outputArgs.front(),
                                                           ValueRange{outputIndex});
          builder.create<scf::YieldOp>(bodyLoc, updated);
        });

    auto stateLoop = rewriter.create<scf::ForOp>(
        loc, zero, stateLength, one, ValueRange{emptyNextState},
        [&](OpBuilder &builder, Location bodyLoc, Value stateIndex, ValueRange stateArgs) {
          Value extendedIndex = builder.create<arith::AddIOp>(bodyLoc, inputLength, stateIndex);
          Value fromState = builder.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::ult,
                                                          extendedIndex, stateLength);
          auto selected = builder.create<scf::IfOp>(
              bodyLoc, TypeRange{op.getInput().getType().getElementType()}, fromState,
              /*withElseRegion=*/true);
          OpBuilder historyBuilder = selected.getThenBodyBuilder();
          Value historyValue = historyBuilder.create<tensor::ExtractOp>(bodyLoc, adaptor.getState(),
                                                                        ValueRange{extendedIndex});
          historyBuilder.create<scf::YieldOp>(bodyLoc, historyValue);
          OpBuilder inputBuilder = selected.getElseBodyBuilder();
          Value inputIndex =
              inputBuilder.create<arith::SubIOp>(bodyLoc, extendedIndex, stateLength);
          Value inputValue = inputBuilder.create<tensor::ExtractOp>(bodyLoc, adaptor.getInput(),
                                                                    ValueRange{inputIndex});
          inputBuilder.create<scf::YieldOp>(bodyLoc, inputValue);
          Value updated = builder.create<tensor::InsertOp>(
              bodyLoc, selected.getResult(0), stateArgs.front(), ValueRange{stateIndex});
          builder.create<scf::YieldOp>(bodyLoc, updated);
        });

    rewriter.replaceOp(op, ValueRange{outputLoop.getResult(0), stateLoop.getResult(0)});
    return success();
  }
};

class SosFilterTdf2OpLowering final : public OpConversionPattern<ondrix::ir::SosFilterTdf2Op> {
public:
  using OpConversionPattern<ondrix::ir::SosFilterTdf2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::SosFilterTdf2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientSections = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value scaleSections = rewriter.create<tensor::DimOp>(loc, adaptor.getScales(), zero);
    Value stateSections = rewriter.create<tensor::DimOp>(loc, adaptor.getState(), zero);

    assertValidSosSectionShape(loc, coefficientSections, scaleSections, stateSections, zero,
                               rewriter);

    Value emptyOutput = createEmptyTensor(loc, op.getOutput().getType(), inputLength, rewriter);
    auto numeric = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    Value coefficientZero = zero;
    Value coefficientOne = one;
    Value coefficientTwo = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value coefficientThree = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value coefficientFour = rewriter.create<arith::ConstantIndexOp>(loc, 4);

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, inputLength, one, ValueRange{emptyOutput, adaptor.getState()},
        [&](OpBuilder &builder, Location sampleLoc, Value sampleIndex, ValueRange sampleArgs) {
          Value sample = builder.create<tensor::ExtractOp>(sampleLoc, adaptor.getInput(),
                                                           ValueRange{sampleIndex});
          auto sectionLoop = builder.create<scf::ForOp>(
              sampleLoc, zero, coefficientSections, one, ValueRange{sample, sampleArgs[1]},
              [&](OpBuilder &sectionBuilder, Location sectionLoc, Value section,
                  ValueRange sectionArgs) {
                auto extractCoefficient = [&](Value column) {
                  return sectionBuilder.create<tensor::ExtractOp>(sectionLoc, adaptor.getCoeffs(),
                                                                  ValueRange{section, column});
                };
                Value scale = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, adaptor.getScales(), ValueRange{section});
                Value b0 = extractCoefficient(coefficientZero);
                Value b1 = extractCoefficient(coefficientOne);
                Value b2 = extractCoefficient(coefficientTwo);
                Value a1 = extractCoefficient(coefficientThree);
                Value a2 = extractCoefficient(coefficientFour);
                Value z1 = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, sectionArgs[1], ValueRange{section, coefficientZero});
                Value z2 = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, sectionArgs[1], ValueRange{section, coefficientOne});

                Value scaled =
                    createFpMultiply(sectionLoc, sectionArgs[0], scale, numeric, sectionBuilder);
                Value output =
                    createFpAccumulatorUpdate(sectionLoc, scaled, b0, z1, numeric, sectionBuilder);
                Value feedback1 = createFpMultiply(sectionLoc, output, a1, numeric, sectionBuilder);
                Value firstTerm = createFpAccumulatorUpdate(sectionLoc, scaled, b1, feedback1,
                                                            numeric, sectionBuilder);
                Value nextZ1 = createFpAdd(sectionLoc, z2, firstTerm, numeric, sectionBuilder);
                Value feedback2 = createFpMultiply(sectionLoc, output, a2, numeric, sectionBuilder);
                Value nextZ2 = createFpAccumulatorUpdate(sectionLoc, scaled, b2, feedback2, numeric,
                                                         sectionBuilder);
                Value stateWithZ1 = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, nextZ1, sectionArgs[1], ValueRange{section, coefficientZero});
                Value nextState = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, nextZ2, stateWithZ1, ValueRange{section, coefficientOne});
                sectionBuilder.create<scf::YieldOp>(sectionLoc, ValueRange{output, nextState});
              });
          Value nextOutput = builder.create<tensor::InsertOp>(
              sampleLoc, sectionLoop.getResult(0), sampleArgs[0], ValueRange{sampleIndex});
          builder.create<scf::YieldOp>(sampleLoc, ValueRange{nextOutput, sectionLoop.getResult(1)});
        });

    rewriter.replaceOp(op, sampleLoop.getResults());
    return success();
  }
};

class SosFilterDf2FixedOpLowering final
    : public OpConversionPattern<ondrix::ir::SosFilterDf2FixedOp> {
public:
  using OpConversionPattern<ondrix::ir::SosFilterDf2FixedOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::SosFilterDf2FixedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value three = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, adaptor.getInput(), zero);
    Value coefficientSections = rewriter.create<tensor::DimOp>(loc, adaptor.getCoeffs(), zero);
    Value scaleSections = rewriter.create<tensor::DimOp>(loc, adaptor.getScales(), zero);
    Value stateSections = rewriter.create<tensor::DimOp>(loc, adaptor.getState(), zero);

    assertValidSosSectionShape(loc, coefficientSections, scaleSections, stateSections, zero,
                               rewriter);

    Value emptyOutput = createEmptyTensor(loc, op.getOutput().getType(), inputLength, rewriter);
    auto createMac = [&](OpBuilder &builder, Location updateLoc, Value accumulator, Value lhs,
                         Value rhs) {
      return builder.create<ondrix::ondsp::MacOp>(updateLoc, op.getAccumulator(), accumulator, lhs,
                                                  rhs, op.getNumeric(), op.getProduct());
    };

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, inputLength, one, ValueRange{emptyOutput, adaptor.getState()},
        [&](OpBuilder &builder, Location sampleLoc, Value sampleIndex, ValueRange sampleArgs) {
          Value sample = builder.create<tensor::ExtractOp>(sampleLoc, adaptor.getInput(),
                                                           ValueRange{sampleIndex});
          auto sectionLoop = builder.create<scf::ForOp>(
              sampleLoc, zero, coefficientSections, one, ValueRange{sample, sampleArgs[1]},
              [&](OpBuilder &sectionBuilder, Location sectionLoc, Value section,
                  ValueRange sectionArgs) {
                auto extractCoefficient = [&](Value column) {
                  return sectionBuilder.create<tensor::ExtractOp>(sectionLoc, adaptor.getCoeffs(),
                                                                  ValueRange{section, column});
                };
                Value scale = sectionBuilder.create<tensor::ExtractOp>(
                    sectionLoc, adaptor.getScales(), ValueRange{section});
                Value b0 = extractCoefficient(zero);
                Value b1 = extractCoefficient(one);
                Value b2 = extractCoefficient(two);
                Value a1 = extractCoefficient(three);
                Value a2 = extractCoefficient(four);
                Value d1 = sectionBuilder.create<tensor::ExtractOp>(sectionLoc, sectionArgs[1],
                                                                    ValueRange{section, zero});
                Value d2 = sectionBuilder.create<tensor::ExtractOp>(sectionLoc, sectionArgs[1],
                                                                    ValueRange{section, one});

                Value stateAccumulator = sectionBuilder.create<ondrix::ondsp::AccZeroOp>(
                    sectionLoc, op.getAccumulator());
                stateAccumulator =
                    createMac(sectionBuilder, sectionLoc, stateAccumulator, sectionArgs[0], scale);
                stateAccumulator = createMac(sectionBuilder, sectionLoc, stateAccumulator, d1, a1);
                stateAccumulator = createMac(sectionBuilder, sectionLoc, stateAccumulator, d2, a2);
                Value nextD1 = sectionBuilder.create<ondrix::ondsp::AccExportOp>(
                    sectionLoc, op.getNumeric().getStorage(), stateAccumulator, op.getNumeric(),
                    op.getStateRounding(), op.getStateOverflow());

                Value outputAccumulator = sectionBuilder.create<ondrix::ondsp::AccZeroOp>(
                    sectionLoc, op.getAccumulator());
                outputAccumulator =
                    createMac(sectionBuilder, sectionLoc, outputAccumulator, nextD1, b0);
                outputAccumulator =
                    createMac(sectionBuilder, sectionLoc, outputAccumulator, d1, b1);
                outputAccumulator =
                    createMac(sectionBuilder, sectionLoc, outputAccumulator, d2, b2);
                Value output = sectionBuilder.create<ondrix::ondsp::AccExportOp>(
                    sectionLoc, op.getNumeric().getStorage(), outputAccumulator, op.getNumeric(),
                    op.getOutputRounding(), op.getOutputOverflow());

                Value stateWithD1 = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, nextD1, sectionArgs[1], ValueRange{section, zero});
                Value nextState = sectionBuilder.create<tensor::InsertOp>(
                    sectionLoc, d1, stateWithD1, ValueRange{section, one});
                sectionBuilder.create<scf::YieldOp>(sectionLoc, ValueRange{output, nextState});
              });
          Value nextOutput = builder.create<tensor::InsertOp>(
              sampleLoc, sectionLoop.getResult(0), sampleArgs[0], ValueRange{sampleIndex});
          builder.create<scf::YieldOp>(sampleLoc, ValueRange{nextOutput, sectionLoop.getResult(1)});
        });

    rewriter.replaceOp(op, sampleLoop.getResults());
    return success();
  }
};

class DotOpLowering final : public OpConversionPattern<ondrix::ir::DotOp> {
public:
  using OpConversionPattern<ondrix::ir::DotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<ShapedType>(op.getLhs().getType())) {
      if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
        Value initial = createReductionZero(op.getLoc(), op.getResult().getType(), rewriter);
        rewriter.replaceOpWithNewOp<ondrix::ondsp::MacOp>(op, op.getResult().getType(), initial,
                                                          adaptor.getLhs(), adaptor.getRhs(), fixed,
                                                          *op.getProduct());
        return success();
      }
      auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
      rewriter.replaceOp(
          op, createScalarFpDot(op.getLoc(), adaptor.getLhs(), adaptor.getRhs(), fp, rewriter));
      return success();
    }

    Value initial = createReductionZero(op.getLoc(), op.getResult().getType(), rewriter);
    auto replacement = rewriter.create<ondrix::ondsp::ReduceMacOp>(
        op.getLoc(), op.getResult().getType(), initial, adaptor.getLhs(), adaptor.getRhs(),
        op.getNumeric(), op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class ButterflyOpLowering final : public OpConversionPattern<ondrix::ir::ButterflyOp> {
public:
  using OpConversionPattern<ondrix::ir::ButterflyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::ButterflyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto layout = dyn_cast<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout)
      return rewriter.notifyMatchFailure(op, "requires an ondsp.cx_layout layout attribute");

    auto replacement = rewriter.create<ondrix::ondsp::CxButterflyOp>(
        op.getLoc(), op.getOut0().getType(), op.getOut1().getType(), adaptor.getA(), adaptor.getB(),
        adaptor.getTwiddle(), layout, op.getNumeric(), op.getProduct(), op.getProductScale(),
        op.getOutputScale());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// One round-half-even signed Q1.15 quantization of a binary64 twiddle
// component through the shared guarded quantizer (the same 2^-20 tie guard
// as the FIR design contract): an admissible value provably quantizes
// exactly like the real-valued cos/sin definition for any evaluation chain
// whose total error stays below the guard — the declared libm/binary64
// budget is more than three orders of magnitude below it. +1.0 saturates to
// 32767 by declared convention; -1.0 is exact. A 50-digit sweep of every
// stage twiddle component for power-of-two sizes up to 1024 shows a
// worst-case margin of 0.0036 LSB, so all supported extents are admissible;
// the guard remains as the fail-closed backstop.
static std::optional<int64_t> quantizeTwiddleComponentQ15(double value) {
  std::optional<ondrix::GuardedQ15Value> quantized = ondrix::quantizeGuardedQ15(value);
  if (!quantized)
    return std::nullopt;
  return quantized->value;
}

static std::optional<uint64_t> getPackedQ15TwiddleBits(ondrix::ir::CfftDirection direction,
                                                       int64_t size, int64_t index) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  double angle = kTwoPi * static_cast<double>(index) / static_cast<double>(size);
  std::optional<int64_t> real = quantizeTwiddleComponentQ15(std::cos(angle));
  double sine = std::sin(angle);
  std::optional<int64_t> imaginary =
      quantizeTwiddleComponentQ15(direction == ondrix::ir::CfftDirection::Forward ? -sine : sine);
  if (!real || !imaginary)
    return std::nullopt;
  return (static_cast<uint64_t>(*imaginary & 0xFFFF) << 16) | static_cast<uint64_t>(*real & 0xFFFF);
}

// One stage twiddle of whichever profile the layout selects. Q15 quantizes a
// binary64 estimate under the tie guard; Q31 reads the offline-frozen table,
// because at Q31 the guard is no longer wide enough to certify an
// in-compiler estimate (see include/ondrix/Support/Q31TwiddleTables.h).
static std::optional<uint64_t> getPackedTwiddleBits(unsigned storageWidth,
                                                    ondrix::ir::CfftDirection direction,
                                                    int64_t size, int64_t index) {
  if (storageWidth == 16)
    return getPackedQ15TwiddleBits(direction, size, index);
  return ondrix::getPackedQ31TwiddleBits(direction == ondrix::ir::CfftDirection::Forward
                                             ? ondrix::Q31TwiddleDirection::Forward
                                             : ondrix::Q31TwiddleDirection::Inverse,
                                         size, index);
}

// Fail-closed availability of every stage twiddle needed by the recursive
// combine of one static extent. The recursion itself may then rely on
// twiddle generation succeeding.
static bool hasAdmissiblePackedTwiddleTables(unsigned storageWidth,
                                             ondrix::ir::CfftDirection direction, int64_t extent) {
  for (int64_t size = 2; size <= extent; size *= 2)
    for (int64_t index = 0; index < size / 2; ++index)
      if (!getPackedTwiddleBits(storageWidth, direction, size, index))
        return false;
  return true;
}

// The real-spectrum lowerings below are packed-Q15 only: their verifiers
// accept no other layout, so they name the layout and let the shared mapping
// supply its widths rather than restating them.
static ondrix::ondsp::PackedComplexProfile getPackedQ15Profile() {
  std::optional<ondrix::ondsp::PackedComplexProfile> profile =
      ondrix::ondsp::getPackedComplexProfile(ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo);
  assert(profile && "the packed Q15 layout must have an executable profile");
  return *profile;
}

static SmallVector<Value>
lowerPackedCfft(Location loc, ArrayRef<Value> inputs, ondrix::ir::CfftDirection direction,
                ondrix::ondsp::PackedComplexProfile profile, ondrix::ondsp::CxLayoutAttr layout,
                Attribute numeric, ondrix::ondsp::ProductAttr product,
                ondrix::ondsp::ScaleAttr productScale, ondrix::ondsp::ScaleAttr outputScale,
                bool vectorizeStaticCfft, ConversionPatternRewriter &rewriter) {
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  auto createPackedTwiddle = [&](uint64_t bits) {
    return rewriter.create<arith::ConstantOp>(
        loc, container,
        rewriter.getIntegerAttr(container, llvm::APInt(profile.containerWidth, bits)));
  };
  auto createButterfly = [&](Value a, Value b, Value twiddle) {
    return rewriter.create<ondrix::ondsp::CxButterflyOp>(loc, container, container, a, b, twiddle,
                                                         layout, numeric, product, productScale,
                                                         outputScale);
  };
  auto buildVector = [&](ArrayRef<Value> values) {
    assert(!values.empty() && "CFFT stage vector must contain at least one lane");
    auto vectorType =
        VectorType::get({static_cast<int64_t>(values.size())}, values.front().getType());
    Value vector = rewriter.create<vector::BroadcastOp>(loc, vectorType, values.front());
    for (auto [index, value] : llvm::enumerate(values.drop_front()))
      vector = rewriter.create<vector::InsertOp>(
          loc, value, vector, ArrayRef<int64_t>{static_cast<int64_t>(index + 1)});
    return vector;
  };

  std::function<SmallVector<Value>(ArrayRef<Value>)> lowerCfft =
      [&](ArrayRef<Value> values) -> SmallVector<Value> {
    if (values.size() == 1)
      return {values.front()};

    SmallVector<Value> evenInputs;
    SmallVector<Value> oddInputs;
    evenInputs.reserve(values.size() / 2);
    oddInputs.reserve(values.size() / 2);
    for (auto [index, value] : llvm::enumerate(values))
      (index % 2 == 0 ? evenInputs : oddInputs).push_back(value);

    SmallVector<Value> even = lowerCfft(evenInputs);
    SmallVector<Value> odd = lowerCfft(oddInputs);
    SmallVector<Value> outputs(values.size());
    if (vectorizeStaticCfft && even.size() > 1) {
      SmallVector<Value> twiddles;
      twiddles.reserve(even.size());
      for (int64_t index = 0, end = even.size(); index < end; ++index) {
        std::optional<uint64_t> twiddleBits =
            getPackedTwiddleBits(profile.storageWidth, direction, values.size(), index);
        assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
        twiddles.push_back(createPackedTwiddle(*twiddleBits));
      }
      Value evenVector = buildVector(even);
      Value oddVector = buildVector(odd);
      Value twiddleVector = buildVector(twiddles);
      auto vectorType = cast<VectorType>(evenVector.getType());
      auto butterfly = rewriter.create<ondrix::ondsp::CxButterflyOp>(
          loc, vectorType, vectorType, evenVector, oddVector, twiddleVector, layout, numeric,
          product, productScale, outputScale);
      for (int64_t index = 0, end = even.size(); index < end; ++index) {
        outputs[index] = rewriter.create<vector::ExtractOp>(loc, butterfly.getOut0(), index);
        outputs[index + end] = rewriter.create<vector::ExtractOp>(loc, butterfly.getOut1(), index);
      }
      return outputs;
    }
    for (int64_t index = 0, end = values.size() / 2; index < end; ++index) {
      std::optional<uint64_t> twiddleBits =
          getPackedTwiddleBits(profile.storageWidth, direction, values.size(), index);
      assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
      auto butterfly = createButterfly(even[index], odd[index], createPackedTwiddle(*twiddleBits));
      outputs[index] = butterfly.getOut0();
      outputs[index + end] = butterfly.getOut1();
    }
    return outputs;
  };
  return lowerCfft(inputs);
}

// Loop-form lowering of the same recursive DIT dataflow with in-memory
// tables. The recursive even/odd combine applied to natural-order input
// computes exactly the butterflies of the iterative algorithm applied to
// bit-reversed input: stage half-sizes H = 1, 2, ..., N/2, butterfly j of
// group g pairing positions g*2H + j and g*2H + j + H under twiddle
// W(2H, j). Every butterfly is the same scalar ondsp.cx_butterfly with the
// same requantization attributes, so each output element passes through the
// identical sequence of quantization boundaries in an equivalent order and
// the two lowerings are bit-identical per element; only the code shape
// changes (loops and constant tables instead of unrolled SSA butterflies).
// This opt-in mode is packed-Q15 only; the caller rejects the packed-Q31
// profile before reaching it.
static Value lowerPackedQ15CfftLoops(Location loc, Value input, int64_t extent,
                                     ondrix::ir::CfftDirection direction,
                                     ondrix::ondsp::CxLayoutAttr layout, Attribute numeric,
                                     ondrix::ondsp::ProductAttr product,
                                     ondrix::ondsp::ScaleAttr productScale,
                                     ondrix::ondsp::ScaleAttr outputScale,
                                     ConversionPatternRewriter &rewriter) {
  IntegerType i32 = rewriter.getI32Type();
  IntegerType i64 = rewriter.getI64Type();
  int64_t stageCount = llvm::Log2_64(extent);

  // twiddles[H + j] = W(2H, j) for H = 1, 2, ..., N/2; index 0 is unused.
  SmallVector<int32_t> twiddleBits(extent, 0);
  for (int64_t half = 1; half < extent; half *= 2)
    for (int64_t index = 0; index < half; ++index) {
      std::optional<uint64_t> bits = getPackedQ15TwiddleBits(direction, 2 * half, index);
      assert(bits && "twiddle admissibility was checked before lowering");
      twiddleBits[half + index] = static_cast<int32_t>(static_cast<uint32_t>(*bits));
    }
  SmallVector<int64_t> bitReversed(extent);
  for (int64_t index = 0; index < extent; ++index) {
    int64_t reversed = 0;
    for (int64_t bit = 0; bit < stageCount; ++bit)
      reversed |= ((index >> bit) & 1) << (stageCount - 1 - bit);
    bitReversed[index] = reversed;
  }
  Value twiddleTable = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(RankedTensorType::get({extent}, i32),
                                  llvm::ArrayRef<int32_t>(twiddleBits)));
  Value reversalTable = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(RankedTensorType::get({extent}, i64),
                                  llvm::ArrayRef<int64_t>(bitReversed)));

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value halfExtent = rewriter.create<arith::ConstantIndexOp>(loc, extent / 2);
  Value stages = rewriter.create<arith::ConstantIndexOp>(loc, stageCount);

  Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, i32);
  auto permuteLoop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{empty},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        Value source64 = builder.create<tensor::ExtractOp>(loc, reversalTable, position);
        Value source = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), source64);
        Value value = builder.create<tensor::ExtractOp>(loc, input, source);
        Value inserted = builder.create<tensor::InsertOp>(loc, value, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });

  auto stageLoop = rewriter.create<scf::ForOp>(
      loc, zero, stages, one, ValueRange{permuteLoop.getResult(0)},
      [&](OpBuilder &builder, Location loc, Value stage, ValueRange stageArgs) {
        Value half = builder.create<arith::ShLIOp>(loc, one, stage);
        auto butterflyLoop = builder.create<scf::ForOp>(
            loc, zero, halfExtent, one, ValueRange{stageArgs.front()},
            [&](OpBuilder &builder, Location loc, Value pair, ValueRange pairArgs) {
              Value group = builder.create<arith::DivUIOp>(loc, pair, half);
              Value phase = builder.create<arith::RemUIOp>(loc, pair, half);
              Value doubled = builder.create<arith::AddIOp>(loc, half, half);
              Value base = builder.create<arith::MulIOp>(loc, group, doubled);
              Value upper = builder.create<arith::AddIOp>(loc, base, phase);
              Value lower = builder.create<arith::AddIOp>(loc, upper, half);
              Value twiddleIndex = builder.create<arith::AddIOp>(loc, half, phase);
              Value a = builder.create<tensor::ExtractOp>(loc, pairArgs.front(), upper);
              Value b = builder.create<tensor::ExtractOp>(loc, pairArgs.front(), lower);
              Value twiddle = builder.create<tensor::ExtractOp>(loc, twiddleTable, twiddleIndex);
              auto butterfly = builder.create<ondrix::ondsp::CxButterflyOp>(
                  loc, i32, i32, a, b, twiddle, layout, numeric, product, productScale,
                  outputScale);
              Value insertUpper = builder.create<tensor::InsertOp>(loc, butterfly.getOut0(),
                                                                   pairArgs.front(), upper);
              Value insertLower =
                  builder.create<tensor::InsertOp>(loc, butterfly.getOut1(), insertUpper, lower);
              builder.create<scf::YieldOp>(loc, insertLower);
            });
        builder.create<scf::YieldOp>(loc, butterflyLoop.getResult(0));
      });
  return stageLoop.getResult(0);
}

static Value canonicalizePackedQ15Real(Location loc, Value packed, OpBuilder &rewriter) {
  Value real = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), packed);
  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), real);
}

static Value conjugatePackedQ15Saturating(Location loc, Value packed, OpBuilder &rewriter) {
  IntegerType i16 = rewriter.getI16Type();
  IntegerType i32 = rewriter.getI32Type();
  Value real = rewriter.create<arith::TruncIOp>(loc, i16, packed);
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 16, 32);
  Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
  Value imaginary = rewriter.create<arith::TruncIOp>(loc, i16, high);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);
  Value minimum = rewriter.create<arith::ConstantIntOp>(loc, -32768, 16);
  Value maximum = rewriter.create<arith::ConstantIntOp>(loc, 32767, 16);
  Value isMinimum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, imaginary, minimum);
  Value negated = rewriter.create<arith::SubIOp>(loc, zero, imaginary);
  Value conjugatedImaginary = rewriter.create<arith::SelectOp>(loc, isMinimum, maximum, negated);
  Value realBits = rewriter.create<arith::ExtUIOp>(loc, i32, real);
  Value imaginaryBits = rewriter.create<arith::ExtUIOp>(loc, i32, conjugatedImaginary);
  Value shiftedImaginary = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, shift);
  return rewriter.create<arith::OrIOp>(loc, shiftedImaginary, realBits);
}

class CfftOpLowering final : public OpConversionPattern<ondrix::ir::CfftOp> {
public:
  CfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::CfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto layout = dyn_cast<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout)
      return rewriter.notifyMatchFailure(op, "requires an ondsp.cx_layout layout attribute");

    std::optional<ondrix::ondsp::PackedComplexProfile> profile =
        ondrix::ondsp::getPackedComplexProfile(layout.getLayout());
    if (!profile)
      return rewriter.notifyMatchFailure(op, "layout has no executable packed complex profile");
    bool isQ15 = profile->storageWidth == 16;
    // Both opt-in code-shape modes still carry hardcoded Q15 tables and i32
    // containers, so they fail closed on the packed-Q31 profile rather than
    // emitting a plausible but unvalidated schedule.
    if (!isQ15 && fftLoops)
      return op.emitOpError("loop-form CFFT lowering supports only the packed Q15 profile");
    if (!isQ15 && vectorizeStaticCfft)
      return op.emitOpError("Vector-batched CFFT lowering supports only the packed Q15 profile");

    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(profile->storageWidth, op.getDirection(), extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    if (fftLoops) {
      Value result = lowerPackedQ15CfftLoops(loc, adaptor.getInput(), extent, op.getDirection(),
                                             layout, op.getNumeric(), op.getProduct(),
                                             op.getProductScale(), op.getOutputScale(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }
    SmallVector<Value> indices;
    SmallVector<Value> inputs;
    indices.reserve(extent);
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      indices.push_back(position);
      inputs.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }
    SmallVector<Value> outputs = lowerPackedCfft(
        loc, inputs, op.getDirection(), *profile, layout, op.getNumeric(), op.getProduct(),
        op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);

    Value result = rewriter.create<tensor::EmptyOp>(loc, op.getResult().getType().getShape(),
                                                    op.getResult().getType().getElementType());
    for (auto [value, position] : llvm::zip_equal(outputs, indices))
      result = rewriter.create<tensor::InsertOp>(loc, value, result, position);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

class RfftOpLowering final : public OpConversionPattern<ondrix::ir::RfftOp> {
public:
  RfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::RfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(getPackedQ15Profile().storageWidth,
                                          ondrix::ir::CfftDirection::Forward, extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    if (fftLoops) {
      IntegerType i32 = rewriter.getI32Type();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, i32);
      auto packLoop = rewriter.create<scf::ForOp>(
          loc, zero, extentValue, one, ValueRange{empty},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value real = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
            Value packed = builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), real);
            Value inserted =
                builder.create<tensor::InsertOp>(loc, packed, iterArgs.front(), position);
            builder.create<scf::YieldOp>(loc, inserted);
          });
      Value spectrum = lowerPackedQ15CfftLoops(
          loc, packLoop.getResult(0), extent, ondrix::ir::CfftDirection::Forward, op.getLayout(),
          op.getNumeric(), op.getProduct(), op.getProductScale(), op.getOutputScale(), rewriter);
      RankedTensorType resultType = op.getResult().getType();
      int64_t binCount = resultType.getDimSize(0);
      Value compact = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultType, spectrum, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(binCount)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
      Value half = rewriter.create<arith::ConstantIndexOp>(loc, extent / 2);
      Value dc = rewriter.create<tensor::ExtractOp>(loc, compact, zero);
      compact = rewriter.create<tensor::InsertOp>(loc, canonicalizePackedQ15Real(loc, dc, rewriter),
                                                  compact, zero);
      Value nyquist = rewriter.create<tensor::ExtractOp>(loc, compact, half);
      compact = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedQ15Real(loc, nyquist, rewriter), compact, half);
      rewriter.replaceOp(op, compact);
      return success();
    }
    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value real = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), real));
    }

    SmallVector<Value> outputs =
        lowerPackedCfft(loc, inputs, ondrix::ir::CfftDirection::Forward, getPackedQ15Profile(),
                        op.getLayout(), op.getNumeric(), op.getProduct(), op.getProductScale(),
                        op.getOutputScale(), vectorizeStaticCfft, rewriter);
    outputs.front() = canonicalizePackedQ15Real(loc, outputs.front(), rewriter);
    outputs[extent / 2] = canonicalizePackedQ15Real(loc, outputs[extent / 2], rewriter);

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0, end = resultType.getDimSize(0); index < end; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      result = rewriter.create<tensor::InsertOp>(loc, outputs[index], result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

class IrfftOpLowering final : public OpConversionPattern<ondrix::ir::IrfftOp> {
public:
  IrfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft, bool fftLoops)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft), fftLoops(fftLoops) {
  }

  LogicalResult matchAndRewrite(ondrix::ir::IrfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getResult().getType().getDimSize(0);
    if (!hasAdmissiblePackedTwiddleTables(getPackedQ15Profile().storageWidth,
                                          ondrix::ir::CfftDirection::Inverse, extent))
      return rewriter.notifyMatchFailure(op, "the stage twiddle table is unavailable");
    int64_t half = extent / 2;
    if (fftLoops) {
      IntegerType i32 = rewriter.getI32Type();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
      Value halfValue = rewriter.create<arith::ConstantIndexOp>(loc, half);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{extent}, i32);
      Value dc = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), zero);
      Value seeded = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedQ15Real(loc, dc, rewriter), empty, zero);
      Value nyquist = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), halfValue);
      seeded = rewriter.create<tensor::InsertOp>(
          loc, canonicalizePackedQ15Real(loc, nyquist, rewriter), seeded, halfValue);
      auto mirrorLoop = rewriter.create<scf::ForOp>(
          loc, one, halfValue, one, ValueRange{seeded},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value bin = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
            Value direct = builder.create<tensor::InsertOp>(loc, bin, iterArgs.front(), position);
            Value mirrored = builder.create<arith::SubIOp>(loc, extentValue, position);
            Value conjugated = conjugatePackedQ15Saturating(loc, bin, builder);
            Value full = builder.create<tensor::InsertOp>(loc, conjugated, direct, mirrored);
            builder.create<scf::YieldOp>(loc, full);
          });
      Value outputs = lowerPackedQ15CfftLoops(
          loc, mirrorLoop.getResult(0), extent, ondrix::ir::CfftDirection::Inverse, op.getLayout(),
          op.getNumeric(), op.getProduct(), op.getProductScale(), op.getOutputScale(), rewriter);
      RankedTensorType resultType = op.getResult().getType();
      Value resultEmpty =
          rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
      auto truncateLoop = rewriter.create<scf::ForOp>(
          loc, zero, extentValue, one, ValueRange{resultEmpty},
          [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
            Value packed = builder.create<tensor::ExtractOp>(loc, outputs, position);
            Value real = builder.create<arith::TruncIOp>(loc, builder.getI16Type(), packed);
            Value inserted =
                builder.create<tensor::InsertOp>(loc, real, iterArgs.front(), position);
            builder.create<scf::YieldOp>(loc, inserted);
          });
      rewriter.replaceOp(op, truncateLoop.getResult(0));
      return success();
    }
    SmallVector<Value> compact;
    compact.reserve(half + 1);
    for (int64_t index = 0; index <= half; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      compact.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }

    SmallVector<Value> spectrum(extent);
    spectrum.front() = canonicalizePackedQ15Real(loc, compact.front(), rewriter);
    spectrum[half] = canonicalizePackedQ15Real(loc, compact[half], rewriter);
    for (int64_t index = 1; index < half; ++index) {
      spectrum[index] = compact[index];
      spectrum[extent - index] = conjugatePackedQ15Saturating(loc, compact[index], rewriter);
    }

    SmallVector<Value> outputs =
        lowerPackedCfft(loc, spectrum, ondrix::ir::CfftDirection::Inverse, getPackedQ15Profile(),
                        op.getLayout(), op.getNumeric(), op.getProduct(), op.getProductScale(),
                        op.getOutputScale(), vectorizeStaticCfft, rewriter);
    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value real = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), outputs[index]);
      result = rewriter.create<tensor::InsertOp>(loc, real, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool vectorizeStaticCfft;
  bool fftLoops;
};

class QuantizeOpLowering final : public OpConversionPattern<ondrix::ir::QuantizeOp> {
public:
  using OpConversionPattern<ondrix::ir::QuantizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<ondrix::ondsp::ConvertOp>(
        op.getLoc(), op.getResult().getType(), adaptor.getInput(), op.getSrc(), op.getDst());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// One complex value of the half-size radix-4 split schedule, carried as two
// i32 SSA components between the explicit ondsp requantization points.
struct SplitComplexValue {
  Value real;
  Value imaginary;
};

class RfftRadix4SplitOpLowering final : public OpConversionPattern<ondrix::ir::RfftRadix4SplitOp> {
public:
  using OpConversionPattern<ondrix::ir::RfftRadix4SplitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::RfftRadix4SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getI32Type();

    auto constant = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
    };
    // Every rounding decision of the schedule is an explicit toward-negative
    // ondsp.round_shift; the i32 carrier width never narrows here, so the
    // scale's overflow mode is unreachable.
    auto floorShift = [&](Value value, unsigned shift) -> Value {
      auto scale = ondrix::ondsp::ScaleAttr::get(context, 0, shift,
                                                 ondrix::ondsp::RoundingMode::TowardNegative,
                                                 ondrix::ondsp::OverflowMode::Wrap, i32);
      return rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, i32, value, scale);
    };
    // The schedule's only reachable clamps: the stage-two saturating
    // combines on 4.12-format values, widened back for exact arithmetic.
    auto saturate16 = [&](Value value) -> Value {
      auto numeric =
          ondrix::ondsp::FixedAttr::get(context, ondrix::ondsp::Signedness::Signed, i16, 12);
      Value clamped = rewriter.create<ondrix::ondsp::SatCastOp>(loc, i16, value, numeric);
      return rewriter.create<arith::ExtSIOp>(loc, i32, clamped);
    };
    auto add = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    };
    auto sub = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::SubIOp>(loc, lhs, rhs);
    };
    auto mul = [&](int64_t coefficient, Value value) -> Value {
      return rewriter.create<arith::MulIOp>(loc, constant(coefficient), value);
    };
    // out = (u + j*v) * (co - j*si): exact i32 cross products, then one
    // sixteen-bit floor shift. The i16 narrowing recorded in the contract is
    // proven exact (P2/P4), so the value legitimately stays in its carrier.
    auto twiddle = [&](Value u, Value v, int64_t co, int64_t si) -> SplitComplexValue {
      Value real = floorShift(add(mul(co, u), mul(si, v)), 16);
      Value imaginary = floorShift(sub(mul(co, v), mul(si, u)), 16);
      return {real, imaginary};
    };

    // Frozen Q15 twiddle pairs (pair index -> co, si); pairs 5, 7, and 8 are
    // never consumed at this length and stay zero placeholders.
    static constexpr int64_t kTwiddles[10][2] = {
        {32767, 0}, {30273, 12539},  {23170, 23170}, {12539, 30273}, {0, 32767},
        {0, 0},     {-23171, 23170}, {0, 0},         {0, 0},         {-30274, -12540}};
    // Frozen split coefficients (bin -> Ar, Ai, Br, Bi); bin 0 is unused.
    static constexpr int64_t kSplitCoefficients[16][4] = {{0, 0, 0, 0},
                                                          {13188, -16069, 19580, 16069},
                                                          {10114, -15137, 22654, 15137},
                                                          {7282, -13623, 25486, 13623},
                                                          {4799, -11585, 27969, 11585},
                                                          {2761, -9102, 30007, 9102},
                                                          {1247, -6270, 31521, 6270},
                                                          {315, -3196, 32453, 3196},
                                                          {0, 0, 32767, 0},
                                                          {315, 3196, 32453, -3196},
                                                          {1247, 6270, 31521, -6270},
                                                          {2761, 9102, 30007, -9102},
                                                          {4799, 11585, 27969, -11585},
                                                          {7282, 13623, 25486, -13623},
                                                          {10114, 15137, 22654, -15137},
                                                          {13188, 16069, 19580, -16069}};

    // View the 32 real Q1.15 samples as 16 complex values.
    SmallVector<Value> samples;
    samples.reserve(32);
    for (int64_t index = 0; index < 32; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value element = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      samples.push_back(rewriter.create<arith::ExtSIOp>(loc, i32, element));
    }
    SmallVector<SplitComplexValue> values(16);
    for (int64_t m = 0; m < 16; ++m)
      values[m] = {samples[2 * m], samples[2 * m + 1]};

    // Stage 1: radix-4 groups (g, g+4, g+8, g+12), twiddle pair index g.
    // The fourteen per-group clamps recorded in the contract are proven
    // inactive (proof P1), so they are omitted without changing any bit.
    for (int64_t group = 0; group < 4; ++group) {
      int64_t a = group, b = group + 4, c = group + 8, d = group + 12;
      Value t0 = floorShift(values[a].real, 2);
      Value t1 = floorShift(values[a].imaginary, 2);
      Value s0 = floorShift(values[c].real, 2);
      Value s1 = floorShift(values[c].imaginary, 2);
      Value sum0 = add(t0, s0);
      Value sum1 = add(t1, s1);
      Value diff0 = sub(t0, s0);
      Value diff1 = sub(t1, s1);
      Value tb0 = floorShift(values[b].real, 2);
      Value tb1 = floorShift(values[b].imaginary, 2);
      Value u0 = floorShift(values[d].real, 2);
      Value u1 = floorShift(values[d].imaginary, 2);
      Value tSum0 = add(tb0, u0);
      Value tSum1 = add(tb1, u1);
      values[a] = {add(floorShift(sum0, 1), floorShift(tSum0, 1)),
                   add(floorShift(sum1, 1), floorShift(tSum1, 1))};
      Value r0 = sub(sum0, tSum0);
      Value r1 = sub(sum1, tSum1);
      values[b] = twiddle(r0, r1, kTwiddles[2 * group][0], kTwiddles[2 * group][1]);
      Value tDiff0 = sub(tb0, u0);
      Value tDiff1 = sub(tb1, u1);
      Value rr0 = sub(diff0, tDiff1);
      Value rr1 = add(diff1, tDiff0);
      Value ss0 = add(diff0, tDiff1);
      Value ss1 = sub(diff1, tDiff0);
      values[c] = twiddle(ss0, ss1, kTwiddles[group][0], kTwiddles[group][1]);
      values[d] = twiddle(rr0, rr1, kTwiddles[3 * group][0], kTwiddles[3 * group][1]);
    }

    // Stage 2: unit-twiddle radix-4 groups (i, i+1, i+2, i+3). Saturating
    // combine first, then independent one-bit floor shifts; these are the
    // schedule's only reachable saturation points.
    for (int64_t group = 0; group < 16; group += 4) {
      SplitComplexValue za = values[group], zb = values[group + 1], zc = values[group + 2],
                        zd = values[group + 3];
      Value r0 = saturate16(add(za.real, zc.real));
      Value r1 = saturate16(add(za.imaginary, zc.imaginary));
      Value s0 = saturate16(sub(za.real, zc.real));
      Value s1 = saturate16(sub(za.imaginary, zc.imaginary));
      Value tSum0 = saturate16(add(zb.real, zd.real));
      Value tSum1 = saturate16(add(zb.imaginary, zd.imaginary));
      Value halfR0 = floorShift(r0, 1);
      Value halfR1 = floorShift(r1, 1);
      Value halfT0 = floorShift(tSum0, 1);
      Value halfT1 = floorShift(tSum1, 1);
      values[group] = {add(halfR0, halfT0), add(halfR1, halfT1)};
      values[group + 1] = {sub(halfR0, halfT0), sub(halfR1, halfT1)};
      Value tDiff0 = saturate16(sub(zb.real, zd.real));
      Value tDiff1 = saturate16(sub(zb.imaginary, zd.imaginary));
      Value halfS0 = floorShift(s0, 1);
      Value halfS1 = floorShift(s1, 1);
      Value halfD0 = floorShift(tDiff0, 1);
      Value halfD1 = floorShift(tDiff1, 1);
      values[group + 2] = {add(halfS0, halfD1), sub(halfS1, halfD0)};
      values[group + 3] = {sub(halfS0, halfD1), add(halfS1, halfD0)};
    }

    // Binary bit reversal: swap the six non-fixed orbits.
    static constexpr int64_t kBitReversalPairs[6][2] = {{1, 8},  {2, 4},  {3, 12},
                                                        {5, 10}, {7, 14}, {11, 13}};
    for (const auto &pair : kBitReversalPairs)
      std::swap(values[pair[0]], values[pair[1]]);

    auto packBin = [&](Value real, Value imaginary) -> Value {
      Value realBits = rewriter.create<arith::ExtUIOp>(
          loc, i32, rewriter.create<arith::TruncIOp>(loc, i16, real));
      Value imaginaryBits = rewriter.create<arith::ExtUIOp>(
          loc, i32, rewriter.create<arith::TruncIOp>(loc, i16, imaginary));
      Value shifted = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, constant(16));
      return rewriter.create<arith::OrIOp>(loc, shifted, realBits);
    };

    // Split stage into compact natural-order bins 0..16; DC and Nyquist
    // imaginary components are exactly zero by construction.
    SmallVector<Value> bins(17);
    bins[0] = packBin(floorShift(add(values[0].real, values[0].imaginary), 1), constant(0));
    bins[16] = packBin(floorShift(sub(values[0].real, values[0].imaginary), 1), constant(0));
    for (int64_t k = 1; k < 16; ++k) {
      const auto &coefficient = kSplitCoefficients[k];
      SplitComplexValue z = values[k];
      SplitComplexValue w = values[16 - k];
      Value accR = add(add(sub(mul(coefficient[0], z.real), mul(coefficient[1], z.imaginary)),
                           mul(coefficient[2], w.real)),
                       mul(coefficient[3], w.imaginary));
      Value accI = add(add(sub(mul(coefficient[3], w.real), mul(coefficient[2], w.imaginary)),
                           mul(coefficient[0], z.imaginary)),
                       mul(coefficient[1], z.real));
      bins[k] = packBin(floorShift(accR, 16), floorShift(accI, 16));
    }

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < 17; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      result = rewriter.create<tensor::InsertOp>(loc, bins[index], result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

static ondrix::ondsp::ScaleAttr getNearestEvenSaturatingShift(MLIRContext *context,
                                                              unsigned shift) {
  return ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/shift,
                                       ondrix::ondsp::RoundingMode::NearestEven,
                                       ondrix::ondsp::OverflowMode::Saturate,
                                       IntegerType::get(context, 16));
}

// Shared table-plus-interpolation lowering for ondrix.sine/cosine. The
// phase offset is 0 for sine and 16384 (one exact quarter turn) for
// cosine; everything else — the tie-guarded 256-entry table, the Q8
// nearest-even interpolation boundary, and the saturating combine — is
// identical by contract.
static LogicalResult lowerQ15Trig(Operation *op, Value input, Value result, Attribute numeric,
                                  int64_t phaseOffset, ConversionPatternRewriter &rewriter) {
  // Compile-time table under the shared guarded quantizer; fail closed if
  // any entry were inadmissible (the committed profile is, by margin
  // evidence, but the guard stays as the backstop).
  SmallVector<int16_t> table;
  table.reserve(256);
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  for (int64_t k = 0; k < 256; ++k) {
    std::optional<ondrix::GuardedQ15Value> entry =
        ondrix::quantizeGuardedQ15(std::sin(kTwoPi * static_cast<double>(k) / 256.0));
    if (!entry)
      return rewriter.notifyMatchFailure(op, "sine table entry is not tie-guard admissible");
    table.push_back(entry->value);
  }

  Location loc = op->getLoc();
  IntegerType i16 = rewriter.getI16Type();
  IntegerType i32 = rewriter.getIntegerType(32);
  int64_t extent = cast<RankedTensorType>(input.getType()).getDimSize(0);
  auto interpolationScale = ondrix::ondsp::ScaleAttr::get(
      rewriter.getContext(), /*preShiftLeft=*/0, /*postShiftRight=*/8,
      ondrix::ondsp::RoundingMode::NearestEven, ondrix::ondsp::OverflowMode::Saturate, i16);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({256}, i16), llvm::ArrayRef<int16_t>(table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value offset = rewriter.create<arith::ConstantIntOp>(loc, phaseOffset, 32);
  Value phaseMask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFF, 32);
  Value indexShift = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
  Value fractionMask = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
  Value one32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{result},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        Value phase = builder.create<tensor::ExtractOp>(loc, input, position);
        // Zero extension reads the raw bits as the unsigned turn phase;
        // the offset add plus mask is the exact modular phase advance.
        Value raw = builder.create<arith::ExtUIOp>(loc, i32, phase);
        Value advanced = builder.create<arith::AddIOp>(loc, raw, offset);
        Value turn = builder.create<arith::AndIOp>(loc, advanced, phaseMask);
        Value tableIndex = builder.create<arith::ShRUIOp>(loc, turn, indexShift);
        Value fraction = builder.create<arith::AndIOp>(loc, turn, fractionMask);
        Value nextRaw = builder.create<arith::AddIOp>(loc, tableIndex, one32);
        Value nextIndex = builder.create<arith::AndIOp>(loc, nextRaw, fractionMask);
        Value lowerIdx =
            builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tableIndex);
        Value upperIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), nextIndex);
        Value lower = builder.create<tensor::ExtractOp>(loc, tableConstant, lowerIdx);
        Value upper = builder.create<tensor::ExtractOp>(loc, tableConstant, upperIdx);
        Value lowerWide = builder.create<arith::ExtSIOp>(loc, i32, lower);
        Value upperWide = builder.create<arith::ExtSIOp>(loc, i32, upper);
        Value delta = builder.create<arith::SubIOp>(loc, upperWide, lowerWide);
        Value product = builder.create<arith::MulIOp>(loc, delta, fraction);
        Value interpolated = builder.create<ondrix::ondsp::RoundShiftOp>(
            loc, builder.getI16Type(), product, interpolationScale);
        Value interpolatedWide = builder.create<arith::ExtSIOp>(loc, i32, interpolated);
        Value combined = builder.create<arith::AddIOp>(loc, lowerWide, interpolatedWide);
        Value saturated =
            builder.create<ondrix::ondsp::SatCastOp>(loc, builder.getI16Type(), combined, numeric);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, saturated, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

template <typename TrigOp, int64_t PhaseOffset>
class TrigOpLowering final : public OpConversionPattern<TrigOp> {
public:
  using OpConversionPattern<TrigOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TrigOp op, typename TrigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getResult().getType();
    Value empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(),
                                                   resultType.getElementType());
    return lowerQ15Trig(op, adaptor.getInput(), empty, op.getNumeric(), PhaseOffset, rewriter);
  }
};

using SineOpLowering = TrigOpLowering<ondrix::ir::SineOp, 0>;
using CosineOpLowering = TrigOpLowering<ondrix::ir::CosineOp, 16384>;

// cos(2*pi*k/N) with the quarter-turn angles evaluated exactly. binary64
// cannot represent pi/2, so libm returns about 1e-16 where the exact cosine
// is zero, and bin N/4 would otherwise not name the bin it says it does.
static double turnCosine(int64_t bin, int64_t extent) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  if (4 * bin % extent == 0) {
    static constexpr double kQuarterTurns[4] = {1.0, 0.0, -1.0, 0.0};
    return kQuarterTurns[(4 * bin / extent) % 4];
  }
  return std::cos(kTwoPi * static_cast<double>(bin) / static_cast<double>(extent));
}

class GoertzelOpLowering final : public OpConversionPattern<ondrix::ir::GoertzelOp> {
public:
  using OpConversionPattern<ondrix::ir::GoertzelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::GoertzelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    double cosine = turnCosine(op.getBin(), extent);
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric()))
      return rewriteFloatingPoint(op, adaptor, fp, extent, cosine, rewriter);
    // The fixed recursion coefficient is one tie-guarded compile-time cosine
    // (the same guarded quantizer as every generated table); inadmissible
    // bins fail closed.
    std::optional<ondrix::GuardedQ15Value> coefficient = ondrix::quantizeGuardedQ15(cosine);
    if (!coefficient)
      return rewriter.notifyMatchFailure(op, "bin coefficient is not tie-guard admissible");
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    Attribute numeric = op.getNumeric();
    ondrix::ondsp::ScaleAttr scale = getNearestEvenSaturatingShift(rewriter.getContext(), 15);
    Value doubledCoefficient =
        rewriter.create<arith::ConstantIntOp>(loc, 2 * int64_t(coefficient->value), 64);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value zero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);

    // m = sat_i16(rhe(2*c*s1 / 2^15)) — the per-step product boundary.
    auto stepProduct = [&](OpBuilder &builder, Location loc, Value s1) -> Value {
      Value wide = builder.create<arith::ExtSIOp>(loc, i64, s1);
      Value product = builder.create<arith::MulIOp>(loc, doubledCoefficient, wide);
      return builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, product, scale);
    };

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{zero16, zero16},
        [&](OpBuilder &builder, Location loc, Value sample, ValueRange states) {
          Value s1 = states[0];
          Value s2 = states[1];
          Value m = stepProduct(builder, loc, s1);
          Value x = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), sample);
          Value xWide = builder.create<arith::ExtSIOp>(loc, i32, x);
          Value mWide = builder.create<arith::ExtSIOp>(loc, i32, m);
          Value s2Wide = builder.create<arith::ExtSIOp>(loc, i32, s2);
          Value sum = builder.create<arith::AddIOp>(loc, xWide, mWide);
          Value combined = builder.create<arith::SubIOp>(loc, sum, s2Wide);
          Value s0 = builder.create<ondrix::ondsp::SatCastOp>(loc, i16, combined, numeric);
          builder.create<scf::YieldOp>(loc, ValueRange{s0, s1});
        });
    Value s1 = sampleLoop.getResult(0);
    Value s2 = sampleLoop.getResult(1);
    Value mFinal = stepProduct(rewriter, loc, s1);

    // energy = s1^2 + s2^2 - m*s2, exact in i64; no further boundary.
    Value s1Wide = rewriter.create<arith::ExtSIOp>(loc, i64, s1);
    Value s2Wide = rewriter.create<arith::ExtSIOp>(loc, i64, s2);
    Value mWide = rewriter.create<arith::ExtSIOp>(loc, i64, mFinal);
    Value s1Square = rewriter.create<arith::MulIOp>(loc, s1Wide, s1Wide);
    Value s2Square = rewriter.create<arith::MulIOp>(loc, s2Wide, s2Wide);
    Value cross = rewriter.create<arith::MulIOp>(loc, mWide, s2Wide);
    Value sum = rewriter.create<arith::AddIOp>(loc, s1Square, s2Square);
    Value energy = rewriter.create<arith::SubIOp>(loc, sum, cross);

    RankedTensorType resultType = op.getEnergy().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value result = rewriter.create<tensor::InsertOp>(loc, energy, empty, zero);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  // Doubling the rounded cosine is exact, so `c2` carries no boundary of its
  // own. Everything but the coefficient product and its input addition is
  // built unflagged, which is what bounds the derivable set the operation
  // description declares.
  static LogicalResult rewriteFloatingPoint(ondrix::ir::GoertzelOp op, OpAdaptor adaptor,
                                            ondrix::ondsp::FpAttr numeric, int64_t extent,
                                            double cosine, ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    Type element = numeric.getFormat();
    Value doubledCoefficient = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(element, 2.0 * static_cast<double>(static_cast<float>(cosine))));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value seed = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(element));
    // The lowering selects the fused member of the fast contract's legal set
    // and emits that choice unflagged. Carrying the declaration onward is not
    // inert: on the pinned toolchain a reassoc-flagged fma is de-fused by the
    // backend, which would hand the schedule choice to codegen.
    ondrix::ondsp::FpAttr macNumeric =
        numeric.getContract() == ondrix::ondsp::FpContractMode::Fast
            ? ondrix::ondsp::FpAttr::get(rewriter.getContext(), numeric.getFormat(),
                                         ondrix::ondsp::FpContractMode::Fma)
            : numeric;

    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{seed, seed},
        [&](OpBuilder &builder, Location loc, Value sample, ValueRange states) {
          Value x = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), sample);
          Value combined =
              createFpAccumulatorUpdate(loc, doubledCoefficient, states[0], x, macNumeric, builder);
          Value next = builder.create<arith::SubFOp>(loc, combined, states[1]);
          builder.create<scf::YieldOp>(loc, ValueRange{next, states[0]});
        });
    Value s1 = sampleLoop.getResult(0);
    Value s2 = sampleLoop.getResult(1);
    Value m = rewriter.create<arith::MulFOp>(loc, doubledCoefficient, s1);
    Value s1Square = rewriter.create<arith::MulFOp>(loc, s1, s1);
    Value s2Square = rewriter.create<arith::MulFOp>(loc, s2, s2);
    Value cross = rewriter.create<arith::MulFOp>(loc, m, s2);
    Value sum = rewriter.create<arith::AddFOp>(loc, s1Square, s2Square);
    Value energy = rewriter.create<arith::SubFOp>(loc, sum, cross);

    RankedTensorType resultType = op.getEnergy().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    rewriter.replaceOp(op, rewriter.create<tensor::InsertOp>(loc, energy, empty, zero).getResult());
    return success();
  }
};

class MatmulOpLowering final : public OpConversionPattern<ondrix::ir::MatmulOp> {
public:
  using OpConversionPattern<ondrix::ir::MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    IntegerType i64 = rewriter.getIntegerType(64);
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    // Exact i64 K-sum per element (|sum| <= 64 * 2^30), one nearest-even
    // saturating boundary. Loop-form over all three dimensions. The f32
    // profile runs the same nest over its declared per-term events and has
    // no boundary after the sum.
    ondrix::ondsp::ScaleAttr scale =
        fp ? ondrix::ondsp::ScaleAttr() : getNearestEvenSaturatingShift(rewriter.getContext(), 15);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rows = rewriter.create<arith::ConstantIndexOp>(loc, lhsType.getDimSize(0));
    Value inner = rewriter.create<arith::ConstantIndexOp>(loc, lhsType.getDimSize(1));
    Value columns = rewriter.create<arith::ConstantIndexOp>(loc, rhsType.getDimSize(1));
    Value seed =
        fp ? rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(fp.getFormat(), 0.0))
                 .getResult()
           : rewriter.create<arith::ConstantIntOp>(loc, 0, 64).getResult();

    RankedTensorType resultType = op.getResult().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    auto rowLoop = rewriter.create<scf::ForOp>(
        loc, zero, rows, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value row, ValueRange rowArgs) {
          auto columnLoop = builder.create<scf::ForOp>(
              loc, zero, columns, one, ValueRange{rowArgs.front()},
              [&](OpBuilder &builder, Location loc, Value column, ValueRange columnArgs) {
                auto accLoop = builder.create<scf::ForOp>(
                    loc, zero, inner, one, ValueRange{seed},
                    [&](OpBuilder &builder, Location loc, Value index, ValueRange accArgs) {
                      Value left = builder.create<tensor::ExtractOp>(loc, adaptor.getLhs(),
                                                                     ValueRange{row, index});
                      Value right = builder.create<tensor::ExtractOp>(loc, adaptor.getRhs(),
                                                                      ValueRange{index, column});
                      if (fp) {
                        Value updated = createFpAccumulatorUpdate(loc, left, right, accArgs.front(),
                                                                  fp, builder);
                        builder.create<scf::YieldOp>(loc, updated);
                        return;
                      }
                      Value leftWide = builder.create<arith::ExtSIOp>(loc, i64, left);
                      Value rightWide = builder.create<arith::ExtSIOp>(loc, i64, right);
                      Value product = builder.create<arith::MulIOp>(loc, leftWide, rightWide);
                      Value sum = builder.create<arith::AddIOp>(loc, accArgs.front(), product);
                      builder.create<scf::YieldOp>(loc, sum);
                    });
                Value element = accLoop.getResult(0);
                if (!fp)
                  element = builder.create<ondrix::ondsp::RoundShiftOp>(loc, builder.getI16Type(),
                                                                        element, scale);
                Value inserted = builder.create<tensor::InsertOp>(loc, element, columnArgs.front(),
                                                                  ValueRange{row, column});
                builder.create<scf::YieldOp>(loc, inserted);
              });
          builder.create<scf::YieldOp>(loc, columnLoop.getResult(0));
        });
    rewriter.replaceOp(op, rowLoop.getResult(0));
    return success();
  }
};

class RmsOpLowering final : public OpConversionPattern<ondrix::ir::RmsOp> {
public:
  using OpConversionPattern<ondrix::ir::RmsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::RmsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    MLIRContext *context = rewriter.getContext();
    RankedTensorType resultTensor = op.getResult().getType();
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric())) {
      Type element = fp.getFormat();
      Value index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value bound = rewriter.create<arith::ConstantIndexOp>(loc, extent);
      Value seed = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(element, 0.0));
      auto sumLoop = rewriter.create<scf::ForOp>(
          loc, index, bound, step, ValueRange{seed},
          [&](OpBuilder &builder, Location bodyLoc, Value position, ValueRange iterArgs) {
            Value value = builder.create<tensor::ExtractOp>(bodyLoc, adaptor.getInput(), position);
            Value updated =
                createFpAccumulatorUpdate(bodyLoc, value, value, iterArgs.front(), fp, builder);
            builder.create<scf::YieldOp>(bodyLoc, updated);
          });
      Value count = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(element, static_cast<double>(extent)));
      Value mean = rewriter.create<arith::DivFOp>(loc, sumLoop.getResult(0), count);
      Value root = rewriter.create<math::SqrtOp>(loc, mean);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, resultTensor.getShape(), element);
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, root, empty, index);
      return success();
    }
    // The exact i64 sum of squares is bounded by 2^42; the nearest-even
    // mean by 2^m is the first boundary (frac 30, at most 2^30, so the
    // declared i32 saturation is unreachable) and the integer root the
    // second. The mean of squares is nonnegative by construction, which
    // establishes the sqrt_fixed value domain structurally.
    auto meanScale = ondrix::ondsp::ScaleAttr::get(
        context, /*preShiftLeft=*/0, /*postShiftRight=*/llvm::Log2_64(extent),
        ondrix::ondsp::RoundingMode::NearestEven, ondrix::ondsp::OverflowMode::Saturate, i32);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value zero64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

    auto accLoop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{zero64},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          Value element = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value wide = builder.create<arith::ExtSIOp>(loc, i64, element);
          Value square = builder.create<arith::MulIOp>(loc, wide, wide);
          Value sum = builder.create<arith::AddIOp>(loc, iterArgs.front(), square);
          builder.create<scf::YieldOp>(loc, sum);
        });
    Value mean =
        rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, i32, accLoop.getResult(0), meanScale);
    Value meanWide = rewriter.create<arith::ExtSIOp>(loc, i64, mean);
    Value root = rewriter.create<ondrix::ondsp::SqrtFixedOp>(loc, rewriter.getI16Type(), meanWide,
                                                             op.getRoundingAttr());

    RankedTensorType resultType = op.getResult().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value result = rewriter.create<tensor::InsertOp>(loc, root, empty, zero);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class GainOpLowering final : public OpConversionPattern<ondrix::ir::GainOp> {
public:
  using OpConversionPattern<ondrix::ir::GainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::GainOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    IntegerType i64 = rewriter.getIntegerType(64);
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    // The product of two Q1.15 values is exact in i64; the single declared
    // boundary is the saturating requantization by 15, under the tie rule
    // the operation declares (the verifier admits nearest_even and
    // nearest_ties_positive). The f32 profile is one multiply per element
    // with no boundary after it. One elementwise loop instead of unrolled
    // inserts: the operation admits extents up to 4096, where a long
    // tensor.insert chain is quadratic in one-shot bufferization.
    ondrix::ondsp::ScaleAttr scale;
    Value gain;
    if (fp) {
      gain = rewriter.create<arith::ConstantOp>(loc, op.getFpGainAttr());
    } else {
      scale = ondrix::ondsp::ScaleAttr::get(
          rewriter.getContext(), /*preShiftLeft=*/0, /*postShiftRight=*/15, *op.getRounding(),
          ondrix::ondsp::OverflowMode::Saturate, rewriter.getI16Type());
      gain = rewriter.create<arith::ConstantIntOp>(loc, op.getGainAttr().getInt(), 64);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);

    RankedTensorType resultType = op.getResult().getType();
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    auto loop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          Value element = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value scaled;
          if (fp) {
            scaled = createFpMultiply(loc, element, gain, fp, builder);
          } else {
            Value wide = builder.create<arith::ExtSIOp>(loc, i64, element);
            Value product = builder.create<arith::MulIOp>(loc, wide, gain);
            scaled = builder.create<ondrix::ondsp::RoundShiftOp>(loc, builder.getI16Type(), product,
                                                                 scale);
          }
          Value inserted =
              builder.create<tensor::InsertOp>(loc, scaled, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class LmsOpLowering final : public OpConversionPattern<ondrix::ir::LmsOp> {
public:
  using OpConversionPattern<ondrix::ir::LmsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::LmsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t samples = op.getInput().getType().getDimSize(0);
    int64_t taps = op.getWeights().getType().getDimSize(0);
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    Attribute numeric = op.getNumeric();
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(numeric);
    Type element = fp ? Type(fp.getFormat()) : Type(i16);
    // Every fixed rounding boundary of the recursion is one nearest-even
    // saturating round_shift by 15; the error and weight updates use
    // explicit saturating casts. The f32 profile has no boundary at any of
    // them. The whole recursion is loop-form either way: the weight state
    // flows sample to sample as an iter_arg.
    ondrix::ondsp::ScaleAttr scale;
    Value mu;
    if (fp) {
      mu = rewriter.create<arith::ConstantOp>(loc, op.getFpStepSizeAttr());
    } else {
      scale = getNearestEvenSaturatingShift(rewriter.getContext(), 15);
      mu = rewriter.create<arith::ConstantIntOp>(loc, op.getStepSizeAttr().getInt(), 64);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value sampleCount = rewriter.create<arith::ConstantIndexOp>(loc, samples);
    Value tapCount = rewriter.create<arith::ConstantIndexOp>(loc, taps);
    Value zeroElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(element));

    // Zero-prehistory tap fetch: x[n - k], 0 for n < k. The prehistory term
    // is evaluated rather than skipped on both profiles.
    auto guardedInput = [&](OpBuilder &builder, Location loc, Value sample, Value tap) -> Value {
      Value offset = builder.create<arith::SubIOp>(loc, sample, tap);
      Value valid = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, offset, zero);
      Value clamped = builder.create<arith::MaxSIOp>(loc, offset, zero);
      Value value = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), clamped);
      return builder.create<arith::SelectOp>(loc, valid, value, zeroElement);
    };

    // Copy the initial weights into a fresh tensor before adapting: the
    // recursion mutates its weight state per sample, and inserting into
    // the function-argument tensor directly would let bufferization adapt
    // the caller's buffer in place.
    Value weightsEmpty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{taps}, element);
    auto copyLoop = rewriter.create<scf::ForOp>(
        loc, zero, tapCount, one, ValueRange{weightsEmpty},
        [&](OpBuilder &builder, Location loc, Value tap, ValueRange iterArgs) {
          Value weight = builder.create<tensor::ExtractOp>(loc, adaptor.getWeights(), tap);
          Value inserted = builder.create<tensor::InsertOp>(loc, weight, iterArgs.front(), tap);
          builder.create<scf::YieldOp>(loc, inserted);
        });

    Value errorsEmpty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{samples}, element);
    if (fp) {
      auto fpLoop = rewriter.create<scf::ForOp>(
          loc, zero, sampleCount, one, ValueRange{copyLoop.getResult(0), errorsEmpty},
          [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs) {
            Value weights = iterArgs[0];
            auto accLoop = builder.create<scf::ForOp>(
                loc, zero, tapCount, one, ValueRange{zeroElement},
                [&](OpBuilder &builder, Location loc, Value tap, ValueRange accArgs) {
                  Value term = guardedInput(builder, loc, sample, tap);
                  Value weight = builder.create<tensor::ExtractOp>(loc, weights, tap);
                  Value updated =
                      createFpAccumulatorUpdate(loc, weight, term, accArgs.front(), fp, builder);
                  builder.create<scf::YieldOp>(loc, updated);
                });
            Value desired = builder.create<tensor::ExtractOp>(loc, adaptor.getDesired(), sample);
            Value error = builder.create<arith::SubFOp>(loc, desired, accLoop.getResult(0));
            Value nextErrors = builder.create<tensor::InsertOp>(loc, error, iterArgs[1], sample);
            Value step = createFpMultiply(loc, mu, error, fp, builder);

            auto updateLoop = builder.create<scf::ForOp>(
                loc, zero, tapCount, one, ValueRange{weights},
                [&](OpBuilder &builder, Location loc, Value tap, ValueRange updateArgs) {
                  Value term = guardedInput(builder, loc, sample, tap);
                  Value weight = builder.create<tensor::ExtractOp>(loc, updateArgs.front(), tap);
                  Value updated = createFpAccumulatorUpdate(loc, step, term, weight, fp, builder);
                  Value inserted =
                      builder.create<tensor::InsertOp>(loc, updated, updateArgs.front(), tap);
                  builder.create<scf::YieldOp>(loc, inserted);
                });
            builder.create<scf::YieldOp>(loc, ValueRange{updateLoop.getResult(0), nextErrors});
          });
      rewriter.replaceOp(op, {fpLoop.getResult(1), fpLoop.getResult(0)});
      return success();
    }

    Value zero64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    auto sampleLoop = rewriter.create<scf::ForOp>(
        loc, zero, sampleCount, one, ValueRange{copyLoop.getResult(0), errorsEmpty},
        [&](OpBuilder &builder, Location loc, Value sample, ValueRange iterArgs) {
          Value weights = iterArgs[0];
          Value errors = iterArgs[1];

          auto accLoop = builder.create<scf::ForOp>(
              loc, zero, tapCount, one, ValueRange{zero64},
              [&](OpBuilder &builder, Location loc, Value tap, ValueRange accArgs) {
                Value term = guardedInput(builder, loc, sample, tap);
                Value termWide = builder.create<arith::ExtSIOp>(loc, i64, term);
                Value weight = builder.create<tensor::ExtractOp>(loc, weights, tap);
                Value weightWide = builder.create<arith::ExtSIOp>(loc, i64, weight);
                Value product = builder.create<arith::MulIOp>(loc, weightWide, termWide);
                Value sum = builder.create<arith::AddIOp>(loc, accArgs.front(), product);
                builder.create<scf::YieldOp>(loc, sum);
              });
          Value output =
              builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, accLoop.getResult(0), scale);

          Value desired = builder.create<tensor::ExtractOp>(loc, adaptor.getDesired(), sample);
          Value desiredWide = builder.create<arith::ExtSIOp>(loc, i32, desired);
          Value outputWide = builder.create<arith::ExtSIOp>(loc, i32, output);
          Value difference = builder.create<arith::SubIOp>(loc, desiredWide, outputWide);
          Value error = builder.create<ondrix::ondsp::SatCastOp>(loc, i16, difference, numeric);
          Value nextErrors = builder.create<tensor::InsertOp>(loc, error, errors, sample);

          Value errorWide = builder.create<arith::ExtSIOp>(loc, i64, error);
          Value stepProduct = builder.create<arith::MulIOp>(loc, mu, errorWide);
          Value step = builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, stepProduct, scale);
          Value stepWide = builder.create<arith::ExtSIOp>(loc, i64, step);

          auto updateLoop = builder.create<scf::ForOp>(
              loc, zero, tapCount, one, ValueRange{weights},
              [&](OpBuilder &builder, Location loc, Value tap, ValueRange updateArgs) {
                Value term = guardedInput(builder, loc, sample, tap);
                Value termWide = builder.create<arith::ExtSIOp>(loc, i64, term);
                Value product = builder.create<arith::MulIOp>(loc, stepWide, termWide);
                Value delta = builder.create<ondrix::ondsp::RoundShiftOp>(loc, i16, product, scale);
                Value weight = builder.create<tensor::ExtractOp>(loc, updateArgs.front(), tap);
                Value weightWide = builder.create<arith::ExtSIOp>(loc, i32, weight);
                Value deltaWide = builder.create<arith::ExtSIOp>(loc, i32, delta);
                Value updated = builder.create<arith::AddIOp>(loc, weightWide, deltaWide);
                Value saturated =
                    builder.create<ondrix::ondsp::SatCastOp>(loc, i16, updated, numeric);
                Value inserted =
                    builder.create<tensor::InsertOp>(loc, saturated, updateArgs.front(), tap);
                builder.create<scf::YieldOp>(loc, inserted);
              });
          builder.create<scf::YieldOp>(loc, ValueRange{updateLoop.getResult(0), nextErrors});
        });
    rewriter.replaceOp(op, {sampleLoop.getResult(1), sampleLoop.getResult(0)});
    return success();
  }
};

class DctOpLowering final : public OpConversionPattern<ondrix::ir::DctOp> {
public:
  using OpConversionPattern<ondrix::ir::DctOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::DctOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getInputNumeric())) {
      Type element = fp.getFormat();
      RankedTensorType outputType = op.getResult().getType();
      Value output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), element);
      for (int64_t k = 0; k < extent; ++k) {
        Value sum;
        for (int64_t n = 0; n < extent; ++n) {
          Value position = rewriter.create<arith::ConstantIndexOp>(loc, n);
          Value value = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value coefficient = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(element, ondrix::getDctCoefficientF32(extent, k, n)));
          sum = sum ? createFpAccumulatorUpdate(loc, value, coefficient, sum, fp, rewriter)
                    : createFpMultiply(loc, value, coefficient, fp, rewriter);
        }
        Value position = rewriter.create<arith::ConstantIndexOp>(loc, k);
        output = rewriter.create<tensor::InsertOp>(loc, sum, output, position);
      }
      rewriter.replaceOp(op, output);
      return success();
    }
    if (!ondrix::hasAdmissibleDctCoefficients(extent))
      return rewriter.notifyMatchFailure(op, "DCT coefficient quantization is not tie-guard "
                                             "admissible");
    IntegerType i64 = rewriter.getIntegerType(64);
    unsigned stageCount = llvm::Log2_64(extent);
    ondrix::ondsp::ScaleAttr scale =
        getNearestEvenSaturatingShift(rewriter.getContext(), 16 + stageCount);

    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value element = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtSIOp>(loc, i64, element));
    }

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t k = 0; k < extent; ++k) {
      // Products and the sum are exact in i64 (|sum| <= N * 2^30 < 2^36),
      // so this reduction has no observable association; the single
      // boundary is the final round_shift.
      Value sum;
      for (int64_t n = 0; n < extent; ++n) {
        int64_t coefficient = *ondrix::getDctCoefficientQ15(extent, k, n);
        Value constant =
            rewriter.create<arith::ConstantOp>(loc, i64, rewriter.getIntegerAttr(i64, coefficient));
        Value product = rewriter.create<arith::MulIOp>(loc, inputs[n], constant);
        sum = sum ? rewriter.create<arith::AddIOp>(loc, sum, product).getResult() : product;
      }
      Value exported =
          rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, rewriter.getI16Type(), sum, scale);
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, k);
      result = rewriter.create<tensor::InsertOp>(loc, exported, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MovingAverageOpLowering final : public OpConversionPattern<ondrix::ir::MovingAverageOp> {
public:
  MovingAverageOpLowering(MLIRContext *context, bool slidingWindowReuse)
      : OpConversionPattern(context), slidingWindowReuse(slidingWindowReuse) {}

  LogicalResult matchAndRewrite(ondrix::ir::MovingAverageOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    int64_t window = op.getWindow();
    IntegerType i64 = rewriter.getIntegerType(64);
    RankedTensorType outputType = op.getResult().getType();
    if (auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric())) {
      // Sliding-window reuse is refused here: the incremental form changes
      // the event graph, and only the exactness of the integer window sums
      // made that value-neutral.
      Type element = fp.getFormat();
      Value count = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(element, static_cast<double>(window)));
      Value output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), element);
      for (int64_t n = 0; n < extent - window + 1; ++n) {
        Value sum;
        for (int64_t k = 0; k < window; ++k) {
          Value position = rewriter.create<arith::ConstantIndexOp>(loc, n + k);
          Value value = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          sum = sum ? createFpAdd(loc, sum, value, fp, rewriter) : value;
        }
        Value mean = rewriter.create<arith::DivFOp>(loc, sum, count);
        Value position = rewriter.create<arith::ConstantIndexOp>(loc, n);
        output = rewriter.create<tensor::InsertOp>(loc, mean, output, position);
      }
      rewriter.replaceOp(op, output);
      return success();
    }
    // A power-of-two window keeps its original round_shift boundary so that
    // profile's lowering stays byte-identical; every other window is the
    // round_div consumer. Both spell the same nearest-even division by K.
    bool windowIsPowerOfTwo = llvm::isPowerOf2_64(window);
    ondrix::ondsp::ScaleAttr scale =
        windowIsPowerOfTwo
            ? getNearestEvenSaturatingShift(rewriter.getContext(), llvm::Log2_64(window))
            : nullptr;

    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value element = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtSIOp>(loc, i64, element));
    }

    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    auto emitMean = [&](Value sum, int64_t n) {
      Value mean;
      if (windowIsPowerOfTwo) {
        mean = rewriter.create<ondrix::ondsp::RoundShiftOp>(loc, rewriter.getI16Type(), sum, scale);
      } else {
        mean = rewriter.create<ondrix::ondsp::RoundDivOp>(
            loc, rewriter.getI16Type(), sum, rewriter.getI64IntegerAttr(window),
            /*pre_shift_left=*/rewriter.getI64IntegerAttr(0),
            ondrix::ondsp::RoundingModeAttr::get(rewriter.getContext(),
                                                 ondrix::ondsp::RoundingMode::NearestEven),
            ondrix::ondsp::OverflowModeAttr::get(rewriter.getContext(),
                                                 ondrix::ondsp::OverflowMode::Saturate));
      }
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, n);
      result = rewriter.create<tensor::InsertOp>(loc, mean, result, position);
    };
    int64_t outputs = extent - window + 1;
    if (slidingWindowReuse) {
      // Incremental running sum — value-neutral only because the window
      // sums are exact with no per-update saturation (the option
      // description carries the argument).
      Value sum = inputs[0];
      for (int64_t i = 1; i < window; ++i)
        sum = rewriter.create<arith::AddIOp>(loc, sum, inputs[i]);
      emitMean(sum, 0);
      for (int64_t n = 1; n < outputs; ++n) {
        Value entered = rewriter.create<arith::AddIOp>(loc, sum, inputs[n + window - 1]);
        sum = rewriter.create<arith::SubIOp>(loc, entered, inputs[n - 1]);
        emitMean(sum, n);
      }
    } else {
      for (int64_t n = 0; n < outputs; ++n) {
        Value sum = inputs[n];
        for (int64_t i = 1; i < window; ++i)
          sum = rewriter.create<arith::AddIOp>(loc, sum, inputs[n + i]);
        emitMean(sum, n);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool slidingWindowReuse;
};

class CxMagnitudeOpLowering final : public OpConversionPattern<ondrix::ir::CxMagnitudeOp> {
public:
  using OpConversionPattern<ondrix::ir::CxMagnitudeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::CxMagnitudeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i64 = rewriter.getIntegerType(64);
    auto roundingAttr =
        ondrix::ondsp::RoundingModeAttr::get(rewriter.getContext(), op.getRounding());

    int64_t extent = op.getInput().getType().getDimSize(0);
    Value shift = rewriter.create<arith::ConstantIntOp>(loc, 16, 32);
    RankedTensorType resultType = op.getResult().getType();
    Value result =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value packed = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      Value real = rewriter.create<arith::TruncIOp>(loc, i16, packed);
      Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
      Value imaginary = rewriter.create<arith::TruncIOp>(loc, i16, high);
      Value realWide = rewriter.create<arith::ExtSIOp>(loc, i64, real);
      Value imaginaryWide = rewriter.create<arith::ExtSIOp>(loc, i64, imaginary);
      Value realSquare = rewriter.create<arith::MulIOp>(loc, realWide, realWide);
      Value imaginarySquare = rewriter.create<arith::MulIOp>(loc, imaginaryWide, imaginaryWide);
      Value sum = rewriter.create<arith::AddIOp>(loc, realSquare, imaginarySquare);
      Value magnitude = rewriter.create<ondrix::ondsp::SqrtFixedOp>(loc, i16, sum, roundingAttr);
      result = rewriter.create<tensor::InsertOp>(loc, magnitude, result, position);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertOndrixToOndspPass final
    : public ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass> {
public:
  using ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass>::ConvertOndrixToOndspBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FirOpLowering, FirFilterOpLowering, FirDecimateOpLowering,
                 FirInterpolateOpLowering, Conv1DOpLowering, FirStreamOpLowering,
                 SosFilterTdf2OpLowering, SosFilterDf2FixedOpLowering, DotOpLowering,
                 ButterflyOpLowering, QuantizeOpLowering, RfftRadix4SplitOpLowering,
                 CxMagnitudeOpLowering, DctOpLowering, GainOpLowering, LmsOpLowering, RmsOpLowering,
                 MatmulOpLowering, GoertzelOpLowering, SineOpLowering, CosineOpLowering>(
        &getContext());
    if (vectorizeStaticCfft && fftLoops) {
      module.emitError("vectorize-static-cfft and fft-loops are mutually exclusive alternative "
                       "FFT lowerings; select at most one");
      return signalPassFailure();
    }
    patterns.add<MovingAverageOpLowering>(&getContext(), slidingWindowReuse);
    patterns.add<CfftOpLowering, RfftOpLowering, IrfftOpLowering>(&getContext(),
                                                                  vectorizeStaticCfft, fftLoops);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect, vector::VectorDialect,
                           ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ir::OndrixDialect>();
    // In the canonical pipeline the operations whose reductions have a direct
    // bufferization stay in contract form through this pass: bufferization
    // lowers them to the reduce_mac loops the schedule stage authorizes over,
    // which the scalar tensor lowering here would preempt.
    if (preserveBufferizableReductions) {
      target.addLegalOp<ondrix::ir::FirFilterOp, ondrix::ir::FirDecimateOp, ondrix::ir::Conv1DOp,
                        ondrix::ir::MatmulOp, ondrix::ir::RmsOp>();
      // The DCT bufferization materializes Q15 coefficient rows as memref
      // globals; the f32 profile has no such table yet and takes the tensor
      // lowering, so it is not preserved.
      target.addDynamicallyLegalOp<ondrix::ir::DctOp>(
          [](ondrix::ir::DctOp op) { return !isa<ondrix::ondsp::FpAttr>(op.getInputNumeric()); });
    }

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}
