#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Support/FirStreamRuntimeShape.h"

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
    return rewriter.create<math::FmaOp>(loc, lhs, rhs, zero, arith::FastMathFlags::fast);
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
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator, arith::FastMathFlags::fast);
  }
  llvm_unreachable("unknown floating-point contract mode");
}

static Value createFpMultiply(Location loc, Value lhs, Value rhs, ondrix::ondsp::FpAttr numeric,
                              OpBuilder &builder) {
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return builder.create<arith::MulFOp>(loc, lhs, rhs, arith::FastMathFlags::fast);
  return builder.create<arith::MulFOp>(loc, lhs, rhs);
}

static Value createFpAdd(Location loc, Value lhs, Value rhs, ondrix::ondsp::FpAttr numeric,
                         OpBuilder &builder) {
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return builder.create<arith::AddFOp>(loc, lhs, rhs, arith::FastMathFlags::fast);
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

    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location outputLoc, Value outputIndex, ValueRange outputArgs) {
          Value inputOrigin = builder.create<arith::MulIOp>(outputLoc, outputIndex, factor);
          Value initial = builder.create<ondrix::ondsp::AccZeroOp>(outputLoc, op.getAccumulator());
          auto tapLoop = builder.create<scf::ForOp>(
              outputLoc, zero, coefficientLength, one, ValueRange{initial},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange tapArgs) {
                Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, inputOrigin, tap);
                Value input =
                    tapBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getInput(), inputIndex);
                Value coefficient =
                    tapBuilder.create<tensor::ExtractOp>(tapLoc, adaptor.getCoeffs(), tap);
                Value updated = tapBuilder.create<ondrix::ondsp::MacOp>(
                    tapLoc, op.getAccumulator(), tapArgs.front(), input, coefficient,
                    op.getNumeric(), op.getProduct());
                tapBuilder.create<scf::YieldOp>(tapLoc, updated);
              });
          Value output = builder.create<ondrix::ondsp::AccExportOp>(
              outputLoc, op.getDst().getStorage(), tapLoop.getResult(0), op.getDst(),
              op.getRounding(), op.getOverflow());
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

    auto outputLoop = rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{adaptor.getInit()},
        [&](OpBuilder &builder, Location outputLoc, Value outputIndex, ValueRange outputArgs) {
          Value initial = builder.create<ondrix::ondsp::AccZeroOp>(outputLoc, op.getAccumulator());
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
                Value updated = thenBuilder.create<ondrix::ondsp::MacOp>(
                    tapLoc, op.getAccumulator(), tapArgs.front(), input, coefficient,
                    op.getNumeric(), op.getProduct());
                thenBuilder.create<scf::YieldOp>(tapLoc, updated);
                OpBuilder elseBuilder = guarded.getElseBodyBuilder();
                elseBuilder.create<scf::YieldOp>(tapLoc, tapArgs.front());
                tapBuilder.create<scf::YieldOp>(tapLoc, guarded.getResult(0));
              });
          Value output = builder.create<ondrix::ondsp::AccExportOp>(
              outputLoc, op.getDst().getStorage(), tapLoop.getResult(0), op.getDst(),
              op.getRounding(), op.getOverflow());
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
// component under the same 2^-20 rounding-tie guard as the FIR design
// contract: a value admissible under the guard provably quantizes exactly
// like the real-valued cos/sin definition regardless of host libm rounding.
// +1.0 saturates to 32767 by declared convention; -1.0 is exact. A 50-digit
// sweep of every stage twiddle component for power-of-two sizes up to 1024
// shows a worst-case margin of 0.0036 LSB, so all supported extents are
// admissible; the guard remains as the fail-closed backstop.
static std::optional<int64_t> quantizeTwiddleComponentQ15(double value) {
  constexpr double kTieGuardLsb = 9.5367431640625e-07; // 2^-20
  double scaled = value * 32768.0;
  double lower = std::floor(scaled);
  double fraction = scaled - lower;
  if (std::fabs(fraction - 0.5) < kTieGuardLsb)
    return std::nullopt;
  int64_t quantized = static_cast<int64_t>(lower) + (fraction > 0.5 ? 1 : 0);
  return std::clamp<int64_t>(quantized, -32768, 32767);
}

static std::optional<uint32_t> getPackedQ15TwiddleBits(ondrix::ir::CfftDirection direction,
                                                       int64_t size, int64_t index) {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  double angle = kTwoPi * static_cast<double>(index) / static_cast<double>(size);
  std::optional<int64_t> real = quantizeTwiddleComponentQ15(std::cos(angle));
  double sine = std::sin(angle);
  std::optional<int64_t> imaginary = quantizeTwiddleComponentQ15(
      direction == ondrix::ir::CfftDirection::Forward ? -sine : sine);
  if (!real || !imaginary)
    return std::nullopt;
  return (static_cast<uint32_t>(*imaginary & 0xFFFF) << 16) |
         static_cast<uint32_t>(*real & 0xFFFF);
}

// Fail-closed admissibility of every stage twiddle needed by the recursive
// combine of one static extent. The recursion itself may then rely on
// twiddle generation succeeding.
static bool hasAdmissiblePackedQ15TwiddleTables(ondrix::ir::CfftDirection direction,
                                                int64_t extent) {
  for (int64_t size = 2; size <= extent; size *= 2)
    for (int64_t index = 0; index < size / 2; ++index)
      if (!getPackedQ15TwiddleBits(direction, size, index))
        return false;
  return true;
}

static SmallVector<Value>
lowerPackedQ15Cfft(Location loc, ArrayRef<Value> inputs, ondrix::ir::CfftDirection direction,
                   ondrix::ondsp::CxLayoutAttr layout, Attribute numeric,
                   ondrix::ondsp::ProductAttr product, ondrix::ondsp::ScaleAttr productScale,
                   ondrix::ondsp::ScaleAttr outputScale, bool vectorizeStaticCfft,
                   ConversionPatternRewriter &rewriter) {
  auto createPackedTwiddle = [&](uint32_t bits) {
    IntegerType i32 = rewriter.getI32Type();
    return rewriter.create<arith::ConstantOp>(loc, i32,
                                              rewriter.getIntegerAttr(i32, llvm::APInt(32, bits)));
  };
  auto createButterfly = [&](Value a, Value b, Value twiddle) {
    return rewriter.create<ondrix::ondsp::CxButterflyOp>(
        loc, rewriter.getI32Type(), rewriter.getI32Type(), a, b, twiddle, layout, numeric, product,
        productScale, outputScale);
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
        std::optional<uint32_t> twiddleBits =
            getPackedQ15TwiddleBits(direction, values.size(), index);
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
      std::optional<uint32_t> twiddleBits =
          getPackedQ15TwiddleBits(direction, values.size(), index);
      assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
      auto butterfly = createButterfly(even[index], odd[index], createPackedTwiddle(*twiddleBits));
      outputs[index] = butterfly.getOut0();
      outputs[index + end] = butterfly.getOut1();
    }
    return outputs;
  };
  return lowerCfft(inputs);
}

static Value canonicalizePackedQ15Real(Location loc, Value packed,
                                       ConversionPatternRewriter &rewriter) {
  Value real = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), packed);
  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), real);
}

static Value conjugatePackedQ15Saturating(Location loc, Value packed,
                                          ConversionPatternRewriter &rewriter) {
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
  CfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft) {}

  LogicalResult matchAndRewrite(ondrix::ir::CfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto layout = dyn_cast<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout)
      return rewriter.notifyMatchFailure(op, "requires an ondsp.cx_layout layout attribute");

    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedQ15TwiddleTables(op.getDirection(), extent))
      return rewriter.notifyMatchFailure(op, "twiddle quantization is not tie-guard admissible");
    SmallVector<Value> indices;
    SmallVector<Value> inputs;
    indices.reserve(extent);
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      indices.push_back(position);
      inputs.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }
    SmallVector<Value> outputs = lowerPackedQ15Cfft(
        loc, inputs, op.getDirection(), layout, op.getNumeric(), op.getProduct(),
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
};

class RfftOpLowering final : public OpConversionPattern<ondrix::ir::RfftOp> {
public:
  RfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft) {}

  LogicalResult matchAndRewrite(ondrix::ir::RfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getInput().getType().getDimSize(0);
    if (!hasAdmissiblePackedQ15TwiddleTables(ondrix::ir::CfftDirection::Forward, extent))
      return rewriter.notifyMatchFailure(op, "twiddle quantization is not tie-guard admissible");
    SmallVector<Value> inputs;
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      Value real = rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
      inputs.push_back(rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), real));
    }

    SmallVector<Value> outputs = lowerPackedQ15Cfft(
        loc, inputs, ondrix::ir::CfftDirection::Forward, op.getLayout(), op.getNumeric(),
        op.getProduct(), op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);
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
};

class IrfftOpLowering final : public OpConversionPattern<ondrix::ir::IrfftOp> {
public:
  IrfftOpLowering(MLIRContext *context, bool vectorizeStaticCfft)
      : OpConversionPattern(context), vectorizeStaticCfft(vectorizeStaticCfft) {}

  LogicalResult matchAndRewrite(ondrix::ir::IrfftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t extent = op.getResult().getType().getDimSize(0);
    if (!hasAdmissiblePackedQ15TwiddleTables(ondrix::ir::CfftDirection::Inverse, extent))
      return rewriter.notifyMatchFailure(op, "twiddle quantization is not tie-guard admissible");
    int64_t half = extent / 2;
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

    SmallVector<Value> outputs = lowerPackedQ15Cfft(
        loc, spectrum, ondrix::ir::CfftDirection::Inverse, op.getLayout(), op.getNumeric(),
        op.getProduct(), op.getProductScale(), op.getOutputScale(), vectorizeStaticCfft, rewriter);
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
      Value magnitude =
          rewriter.create<ondrix::ondsp::SqrtFixedOp>(loc, i16, sum, roundingAttr);
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
                 CxMagnitudeOpLowering>(&getContext());
    patterns.add<CfftOpLowering, RfftOpLowering, IrfftOpLowering>(&getContext(),
                                                                  vectorizeStaticCfft);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect, vector::VectorDialect,
                           ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ir::OndrixDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}
