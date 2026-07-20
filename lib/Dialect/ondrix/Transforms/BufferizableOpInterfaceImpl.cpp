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

static Value createInitialAccumulator(FirFilterOp op, OpBuilder &builder, Location loc) {
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

static Value exportFirSample(FirFilterOp op, Value accumulator, OpBuilder &builder, Location loc) {
  if (!isa<ondrix::ondsp::FixedAttr>(op.getNumeric()))
    return accumulator;
  return builder.create<ondrix::ondsp::AccExportOp>(loc, op.getDst()->getStorage(), accumulator,
                                                    *op.getDst(), *op.getRounding(),
                                                    *op.getOverflow());
}

static Value createReducedFirSample(FirFilterOp op, Value window, Value coefficients,
                                    OpBuilder &builder, Location loc) {
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
      // The output tiler proves the complete dynamic shape once before its
      // loop. Untiled operations still need the standalone diagnostic guard.
      if (!op.getOutputOrigin())
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

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, OndrixDialect *) {
    FirFilterOp::attachInterface<FirFilterOpInterface>(*context);

    // Bufferization materializes these dialects even when the input module
    // contains only tensor-form Ondrix operations.
    context->loadDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                         memref::MemRefDialect, scf::SCFDialect, ondrix::ondsp::OndspDialect>();
  });
}

} // namespace ondrix::ir
