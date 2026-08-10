#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/FirStreamRuntimeShape.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

using namespace mlir;
using namespace ondrix::conversion;

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
    return ondrix::ondsp::consumeFastPermission(rewriter.create<math::FmaOp>(loc, lhs, rhs, zero),
                                                ondrix::ondsp::FastPermission::FuseMultiplyAdd);
  }
  }
  llvm_unreachable("unknown floating-point contract mode");
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
            scaled = createFpMultiply(loc, element, gain, builder);
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
          sum = sum ? createFpAdd(loc, sum, value, rewriter) : value;
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

} // namespace

void ondrix::conversion::populateOndrixFirFamilyLoweringPatterns(RewritePatternSet &patterns,
                                                                 bool slidingWindowReuse) {
  MLIRContext *context = patterns.getContext();
  patterns.add<FirOpLowering, FirFilterOpLowering, FirDecimateOpLowering, FirInterpolateOpLowering,
               Conv1DOpLowering, FirStreamOpLowering, DotOpLowering, MatmulOpLowering,
               RmsOpLowering, GainOpLowering>(context);
  patterns.add<MovingAverageOpLowering>(context, slidingWindowReuse);
}
