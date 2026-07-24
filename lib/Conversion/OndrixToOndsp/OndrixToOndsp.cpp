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

#include <cassert>
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
    SmallVector<Value> indices;
    SmallVector<Value> inputs;
    indices.reserve(extent);
    inputs.reserve(extent);
    for (int64_t index = 0; index < extent; ++index) {
      Value position = rewriter.create<arith::ConstantIndexOp>(loc, index);
      indices.push_back(position);
      inputs.push_back(rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), position));
    }

    auto createPackedTwiddle = [&](uint32_t bits) {
      IntegerType i32 = rewriter.getI32Type();
      return rewriter.create<arith::ConstantOp>(
          loc, i32, rewriter.getIntegerAttr(i32, llvm::APInt(32, bits)));
    };
    auto getTwiddleBits = [&](int64_t size, int64_t index) -> std::optional<uint32_t> {
      constexpr uint32_t one = 0x00007fffU;
      constexpr uint32_t minusJ = 0x80000000U;
      constexpr uint32_t plusJ = 0x7fff0000U;
      if (size == 2 && index == 0)
        return one;
      if (size == 4) {
        constexpr uint32_t forwardValues[] = {one, minusJ};
        constexpr uint32_t inverseValues[] = {one, plusJ};
        return op.getDirection() == ondrix::ir::CfftDirection::Forward ? forwardValues[index]
                                                                       : inverseValues[index];
      }
      if (size == 8) {
        constexpr uint32_t forwardValues[] = {one, 0xa57e5a82U, minusJ, 0xa57ea57eU};
        constexpr uint32_t inverseValues[] = {one, 0x5a825a82U, plusJ, 0x5a82a57eU};
        return op.getDirection() == ondrix::ir::CfftDirection::Forward ? forwardValues[index]
                                                                       : inverseValues[index];
      }
      return std::nullopt;
    };
    auto createButterfly = [&](Value a, Value b, Value twiddle) {
      return rewriter.create<ondrix::ondsp::CxButterflyOp>(
          loc, rewriter.getI32Type(), rewriter.getI32Type(), a, b, twiddle, layout, op.getNumeric(),
          op.getProduct(), op.getProductScale(), op.getOutputScale());
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
          std::optional<uint32_t> twiddleBits = getTwiddleBits(values.size(), index);
          assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
          twiddles.push_back(createPackedTwiddle(*twiddleBits));
        }
        Value evenVector = buildVector(even);
        Value oddVector = buildVector(odd);
        Value twiddleVector = buildVector(twiddles);
        auto vectorType = cast<VectorType>(evenVector.getType());
        auto butterfly = rewriter.create<ondrix::ondsp::CxButterflyOp>(
            loc, vectorType, vectorType, evenVector, oddVector, twiddleVector, layout,
            op.getNumeric(), op.getProduct(), op.getProductScale(), op.getOutputScale());
        for (int64_t index = 0, end = even.size(); index < end; ++index) {
          outputs[index] = rewriter.create<vector::ExtractOp>(loc, butterfly.getOut0(), index);
          outputs[index + end] =
              rewriter.create<vector::ExtractOp>(loc, butterfly.getOut1(), index);
        }
        return outputs;
      }
      for (int64_t index = 0, end = values.size() / 2; index < end; ++index) {
        std::optional<uint32_t> twiddleBits = getTwiddleBits(values.size(), index);
        assert(twiddleBits && "verified CFFT extent must have a static twiddle table");
        auto butterfly =
            createButterfly(even[index], odd[index], createPackedTwiddle(*twiddleBits));
        outputs[index] = butterfly.getOut0();
        outputs[index + end] = butterfly.getOut1();
      }
      return outputs;
    };
    SmallVector<Value> outputs = lowerCfft(inputs);

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

class ConvertOndrixToOndspPass final
    : public ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass> {
public:
  using ondrix::impl::ConvertOndrixToOndspBase<ConvertOndrixToOndspPass>::ConvertOndrixToOndspBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FirOpLowering, FirFilterOpLowering, Conv1DOpLowering, FirStreamOpLowering,
                 SosFilterTdf2OpLowering, SosFilterDf2FixedOpLowering, DotOpLowering,
                 ButterflyOpLowering, QuantizeOpLowering>(&getContext());
    patterns.add<CfftOpLowering>(&getContext(), vectorizeStaticCfft);

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
