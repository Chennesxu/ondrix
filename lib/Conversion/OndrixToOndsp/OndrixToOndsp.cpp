#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

    Value hasCoefficients =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
    rewriter.create<cf::AssertOp>(
        loc, hasCoefficients,
        rewriter.getStringAttr("FIR stream requires at least one coefficient"));
    Value expectedCoefficientLength = rewriter.create<arith::AddIOp>(loc, stateLength, one);
    Value stateMatches = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, expectedCoefficientLength, coefficientLength);
    rewriter.create<cf::AssertOp>(
        loc, stateMatches,
        rewriter.getStringAttr("FIR stream state length must equal coefficient length minus one"));
    Value extendedLength = rewriter.create<arith::AddIOp>(loc, stateLength, inputLength);
    Value extendedLengthFits =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, extendedLength, inputLength);
    rewriter.create<cf::AssertOp>(
        loc, extendedLengthFits,
        rewriter.getStringAttr("FIR stream history and input exceed the indexable extent range"));

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
        adaptor.getTwiddle(), layout, op.getNumeric(),
        op.getProduct().value_or(ondrix::ondsp::ProductAttr()),
        op.getScale().value_or(ondrix::ondsp::ScaleAttr()), op.getTrivialTwiddle());
    rewriter.replaceOp(op, replacement);
    return success();
  }
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
    patterns.add<FirOpLowering, FirFilterOpLowering, FirStreamOpLowering, DotOpLowering,
                 ButterflyOpLowering, QuantizeOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, tensor::TensorDialect, ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ir::OndrixDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}
