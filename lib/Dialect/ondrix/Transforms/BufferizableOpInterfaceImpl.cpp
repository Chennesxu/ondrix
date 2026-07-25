#include "ondrix/Dialect/ondrix/Transforms/BufferizableOpInterfaceImpl.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace ondrix::ir {
namespace {

template <typename OpTy>
static Value createInitialAccumulator(OpTy op, OpBuilder &builder, Location loc) {
  if (isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return builder.create<ondrix::ondsp::AccZeroOp>(loc, *op.getAccumulator());
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  return builder.create<arith::ConstantOp>(loc, fp.getFormat(),
                                           builder.getZeroAttr(fp.getFormat()));
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

template <typename OpTy>
static Value exportFirSample(OpTy op, Value accumulator, OpBuilder &builder, Location loc) {
  if (!isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return accumulator;
  return builder.create<ondrix::ondsp::AccExportOp>(loc, op.getDst()->getStorage(), accumulator,
                                                    *op.getDst(), *op.getRounding(),
                                                    *op.getOverflow());
}

template <typename OpTy>
static Value createReducedFirSample(OpTy op, Value window, Value coefficients, OpBuilder &builder,
                                    Location loc) {
  Value initial = createInitialAccumulator(op, builder, loc);
  Value reduced = builder.create<ondrix::ondsp::ReduceMacOp>(
      loc, initial.getType(), initial, window, coefficients, op.getNumeric(),
      op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
  return exportFirSample(op, reduced, builder, loc);
}

static Value createGuardedFullFirSample(FirFilterOp op, Value input, Value coefficients,
                                        Value outputIndex, Value inputLength,
                                        Value coefficientLength, Value leftPadding, Value zero,
                                        Value one, OpBuilder &builder, Location loc) {
  Value initial = createInitialAccumulator(op, builder, loc);
  auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  Value outputBeforeLeft =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, outputIndex, leftPadding);
  Value leftDeficit = builder.create<arith::SubIOp>(loc, leftPadding, outputIndex);
  Value firstValidTap = builder.create<arith::SelectOp>(loc, outputBeforeLeft, leftDeficit, zero);
  Value inputBase = builder.create<arith::SubIOp>(loc, outputIndex, leftPadding);
  auto tapLoop = builder.create<scf::ForOp>(
      loc, zero, coefficientLength, one, ValueRange{initial},
      [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange accumulatorArgs) {
        Value pastLeftPadding =
            tapBuilder.create<arith::CmpIOp>(tapLoc, arith::CmpIPredicate::uge, tap, firstValidTap);
        Value inputIndex = tapBuilder.create<arith::AddIOp>(tapLoc, inputBase, tap);
        Value beforeRightPadding = tapBuilder.create<arith::CmpIOp>(
            tapLoc, arith::CmpIPredicate::ult, inputIndex, inputLength);
        Value inBounds =
            tapBuilder.create<arith::AndIOp>(tapLoc, pastLeftPadding, beforeRightPadding);
        auto guarded = tapBuilder.create<scf::IfOp>(
            tapLoc, TypeRange{accumulatorArgs.front().getType()}, inBounds,
            /*withElseRegion=*/true);

        OpBuilder thenBuilder = guarded.getThenBodyBuilder();
        Value inputValue = thenBuilder.create<memref::LoadOp>(tapLoc, input, inputIndex);
        Value coefficient = thenBuilder.create<memref::LoadOp>(tapLoc, coefficients, tap);
        Value updated;
        if (fixed) {
          updated = thenBuilder.create<ondrix::ondsp::MacOp>(tapLoc, *op.getAccumulator(),
                                                             accumulatorArgs.front(), inputValue,
                                                             coefficient, fixed, *op.getProduct());
        } else {
          updated = createFpAccumulatorUpdate(tapLoc, inputValue, coefficient,
                                              accumulatorArgs.front(), fp, thenBuilder);
        }
        thenBuilder.create<scf::YieldOp>(tapLoc, updated);

        OpBuilder elseBuilder = guarded.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(tapLoc, accumulatorArgs.front());
        tapBuilder.create<scf::YieldOp>(tapLoc, guarded.getResult(0));
      });
  return exportFirSample(op, tapLoop.getResult(0), builder, loc);
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

static Value createReducedFirDecimateSample(FirDecimateOp op, Value window, Value coefficients,
                                            OpBuilder &builder, Location loc) {
  Value initial = builder.create<ondrix::ondsp::AccZeroOp>(loc, op.getAccumulator());
  Value reduced = builder.create<ondrix::ondsp::ReduceMacOp>(
      loc, initial.getType(), initial, window, coefficients, op.getNumeric(), op.getProduct());
  return builder.create<ondrix::ondsp::AccExportOp>(
      loc, op.getDst().getStorage(), reduced, op.getDst(), op.getRounding(), op.getOverflow());
}

struct FirFilterOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FirFilterOpInterface, FirFilterOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto fir = cast<FirFilterOp>(op);
    return !fir.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<FirFilterOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> coefficients = getBuffer(rewriter, op.getCoeffs(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(coefficients) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value coefficientLength = rewriter.create<memref::DimOp>(loc, *coefficients, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    Value globalOutputOrigin = op.getOutputOrigin() ? op.getOutputOrigin() : zero;
    SmallVector<OpFoldResult> coefficientOffsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> coefficientSizes{coefficientLength};
    SmallVector<OpFoldResult> coefficientStrides{rewriter.getIndexAttr(1)};
    Value coefficientView = rewriter.create<memref::SubViewOp>(
        loc, *coefficients, coefficientOffsets, coefficientSizes, coefficientStrides);

    auto createValidRange = [&](Value lower, Value upper) {
      rewriter.create<scf::ForOp>(
          loc, lower, upper, one, ValueRange{},
          [&](OpBuilder &builder, Location bodyLoc, Value localOutputIndex, ValueRange) {
            Value globalOutputIndex = localOutputIndex;
            if (op.getOutputOrigin())
              globalOutputIndex =
                  builder.create<arith::AddIOp>(bodyLoc, globalOutputOrigin, localOutputIndex);
            Value inputOffset = globalOutputIndex;
            if (op.getBoundary() == FirBoundaryMode::Full) {
              Value leftPadding = builder.create<arith::SubIOp>(bodyLoc, coefficientLength, one);
              inputOffset = builder.create<arith::SubIOp>(bodyLoc, globalOutputIndex, leftPadding);
            }
            SmallVector<OpFoldResult> offsets{inputOffset};
            SmallVector<OpFoldResult> sizes{coefficientLength};
            SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
            Value window =
                builder.create<memref::SubViewOp>(bodyLoc, *input, offsets, sizes, strides);
            Value sample = createReducedFirSample(op, window, coefficientView, builder, bodyLoc);
            builder.create<memref::StoreOp>(bodyLoc, sample, *output, localOutputIndex);
            builder.create<scf::YieldOp>(bodyLoc);
          });
    };

    if (op.getBoundary() == FirBoundaryMode::Valid) {
      createValidRange(zero, outputLength);
    } else {
      if (op.getOutputOrigin())
        assertFullFirFilterTileShape(loc, inputLength, coefficientLength, outputLength,
                                     globalOutputOrigin, zero, one, rewriter);
      else
        assertFullFirFilterShape(loc, inputLength, coefficientLength, outputLength, zero, one,
                                 rewriter);
      Value leftPadding = rewriter.create<arith::SubIOp>(loc, coefficientLength, one);
      auto createGuardedRange = [&](Value lower, Value upper) {
        rewriter.create<scf::ForOp>(
            loc, lower, upper, one, ValueRange{},
            [&](OpBuilder &builder, Location bodyLoc, Value localOutputIndex, ValueRange) {
              Value globalOutputIndex = localOutputIndex;
              if (op.getOutputOrigin())
                globalOutputIndex =
                    builder.create<arith::AddIOp>(bodyLoc, globalOutputOrigin, localOutputIndex);
              Value sample = createGuardedFullFirSample(
                  op, *input, coefficientView, globalOutputIndex, inputLength, coefficientLength,
                  leftPadding, zero, one, builder, bodyLoc);
              builder.create<memref::StoreOp>(bodyLoc, sample, *output, localOutputIndex);
              builder.create<scf::YieldOp>(bodyLoc);
            });
      };

      Value globalOutputEnd = outputLength;
      if (op.getOutputOrigin())
        globalOutputEnd = rewriter.create<arith::AddIOp>(loc, globalOutputOrigin, outputLength);

      Value leftEnd = rewriter.create<arith::MinUIOp>(loc, globalOutputEnd, leftPadding);
      leftEnd = rewriter.create<arith::MaxUIOp>(loc, leftEnd, globalOutputOrigin);
      Value localLeftEnd = rewriter.create<arith::SubIOp>(loc, leftEnd, globalOutputOrigin);
      createGuardedRange(zero, localLeftEnd);

      Value interiorStart = rewriter.create<arith::MaxUIOp>(loc, globalOutputOrigin, leftPadding);
      Value interiorEnd = rewriter.create<arith::MinUIOp>(loc, globalOutputEnd, inputLength);
      interiorEnd = rewriter.create<arith::MaxUIOp>(loc, interiorEnd, interiorStart);
      Value localInteriorStart =
          rewriter.create<arith::SubIOp>(loc, interiorStart, globalOutputOrigin);
      Value localInteriorEnd = rewriter.create<arith::SubIOp>(loc, interiorEnd, globalOutputOrigin);
      createValidRange(localInteriorStart, localInteriorEnd);

      Value rightBoundary = rewriter.create<arith::MaxUIOp>(loc, inputLength, leftPadding);
      Value rightStart = rewriter.create<arith::MaxUIOp>(loc, globalOutputOrigin, rightBoundary);
      rightStart = rewriter.create<arith::MinUIOp>(loc, rightStart, globalOutputEnd);
      Value localRightStart = rewriter.create<arith::SubIOp>(loc, rightStart, globalOutputOrigin);
      createGuardedRange(localRightStart, outputLength);
    }

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

struct FirDecimateOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FirDecimateOpInterface, FirDecimateOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto decimate = cast<FirDecimateOp>(op);
    return !decimate.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<FirDecimateOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> coefficients = getBuffer(rewriter, op.getCoeffs(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(coefficients) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value factor =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getFactorAttr().getValue().getSExtValue());
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value coefficientLength = rewriter.create<memref::DimOp>(loc, *coefficients, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    assertValidFirDecimateShape(loc, inputLength, coefficientLength, outputLength, factor, zero,
                                one, rewriter);

    Value coefficientView = rewriter.create<memref::SubViewOp>(
        loc, *coefficients, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{coefficientLength},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
    rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange) {
          Value inputOffset = builder.create<arith::MulIOp>(bodyLoc, outputIndex, factor);
          Value inputWindow = builder.create<memref::SubViewOp>(
              bodyLoc, *input, ArrayRef<OpFoldResult>{inputOffset},
              ArrayRef<OpFoldResult>{coefficientLength},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
          Value sample =
              createReducedFirDecimateSample(op, inputWindow, coefficientView, builder, bodyLoc);
          builder.create<memref::StoreOp>(bodyLoc, sample, *output, outputIndex);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

struct Conv1DOpInterface
    : public DstBufferizableOpInterfaceExternalModel<Conv1DOpInterface, Conv1DOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto conv = cast<Conv1DOp>(op);
    return !conv.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<Conv1DOp>(operation);
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> kernel = getBuffer(rewriter, op.getKernel(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(kernel) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<memref::DimOp>(loc, *input, zero);
    Value kernelLength = rewriter.create<memref::DimOp>(loc, *kernel, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    assertValidConv1DShape(loc, inputLength, kernelLength, outputLength, zero, one, rewriter);

    OpFoldResult kernelOffset = rewriter.getIndexAttr(0);
    OpFoldResult kernelStride = rewriter.getIndexAttr(1);
    if (op.getMode() == Conv1DMode::Convolution) {
      kernelOffset = rewriter.create<arith::SubIOp>(loc, kernelLength, one).getResult();
      kernelStride = rewriter.getIndexAttr(-1);
    }
    Value kernelView = rewriter.create<memref::SubViewOp>(
        loc, *kernel, ArrayRef<OpFoldResult>{kernelOffset}, ArrayRef<OpFoldResult>{kernelLength},
        ArrayRef<OpFoldResult>{kernelStride});

    rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange) {
          Value inputWindow = builder.create<memref::SubViewOp>(
              bodyLoc, *input, ArrayRef<OpFoldResult>{outputIndex},
              ArrayRef<OpFoldResult>{kernelLength},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
          Value sample = createReducedFirSample(op, inputWindow, kernelView, builder, bodyLoc);
          builder.create<memref::StoreOp>(bodyLoc, sample, *output, outputIndex);
          builder.create<scf::YieldOp>(bodyLoc);
        });

    replaceOpWithBufferizedValues(rewriter, op, *output);
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, OndrixDialect *) {
    FirFilterOp::attachInterface<FirFilterOpInterface>(*context);
    FirDecimateOp::attachInterface<FirDecimateOpInterface>(*context);
    Conv1DOp::attachInterface<Conv1DOpInterface>(*context);

    // Bufferization materializes these dialects even when the input module
    // contains only tensor-form Ondrix operations.
    context->loadDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                         memref::MemRefDialect, scf::SCFDialect, ondrix::ondsp::OndspDialect>();
  });
}

} // namespace ondrix::ir
