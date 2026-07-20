#include "ondrix/Dialect/ondrix/Transforms/BufferizableOpInterfaceImpl.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace ondrix::ir {
namespace {

struct FirFilterOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FirFilterOpInterface, FirFilterOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &) const {
    auto fir = cast<FirFilterOp>(op);
    return !fir.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<FirFilterOp>(operation);
    if (op.getBoundary() != FirBoundaryMode::Valid)
      return op.emitOpError("full FIR bufferization is not implemented");
    FailureOr<Value> input = getBuffer(rewriter, op.getInput(), options);
    FailureOr<Value> coefficients = getBuffer(rewriter, op.getCoeffs(), options);
    FailureOr<Value> output = getBuffer(rewriter, op.getInit(), options);
    if (failed(input) || failed(coefficients) || failed(output))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value coefficientLength = rewriter.create<memref::DimOp>(loc, *coefficients, zero);
    Value outputLength = rewriter.create<memref::DimOp>(loc, *output, zero);
    SmallVector<OpFoldResult> coefficientOffsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> coefficientSizes{coefficientLength};
    SmallVector<OpFoldResult> coefficientStrides{rewriter.getIndexAttr(1)};
    Value coefficientView = rewriter.create<memref::SubViewOp>(
        loc, *coefficients, coefficientOffsets, coefficientSizes, coefficientStrides);

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    rewriter.create<scf::ForOp>(
        loc, zero, outputLength, one, ValueRange{},
        [&](OpBuilder &builder, Location bodyLoc, Value outputIndex, ValueRange) {
          SmallVector<OpFoldResult> offsets{outputIndex};
          SmallVector<OpFoldResult> sizes{coefficientLength};
          SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
          Value window =
              builder.create<memref::SubViewOp>(bodyLoc, *input, offsets, sizes, strides);

          Type accumulatorType = fixed ? Type(*op.getAccumulator()) : Type(fp.getFormat());
          Value initial;
          if (fixed) {
            initial = builder.create<ondrix::ondsp::AccZeroOp>(bodyLoc, accumulatorType);
          } else {
            initial = builder.create<arith::ConstantOp>(bodyLoc, accumulatorType,
                                                        builder.getZeroAttr(accumulatorType));
          }

          Value reduced = builder.create<ondrix::ondsp::ReduceMacOp>(
              bodyLoc, accumulatorType, initial, window, coefficientView, op.getNumeric(),
              op.getProduct().value_or(ondrix::ondsp::ProductAttr()));
          Value sample = reduced;
          if (fixed) {
            sample = builder.create<ondrix::ondsp::AccExportOp>(
                bodyLoc, op.getDst()->getStorage(), reduced, *op.getDst(), *op.getRounding(),
                *op.getOverflow());
          }
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

    // Bufferization materializes these dialects even when the input module
    // contains only tensor-form Ondrix operations.
    context->loadDialect<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                         ondrix::ondsp::OndspDialect>();
  });
}

} // namespace ondrix::ir
