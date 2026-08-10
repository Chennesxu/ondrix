#include "ondrix/Transforms/Passes.h"

#include "ondrix/Conversion/Utils/ReductionUtils.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_SCALARIZEONDSPFIXEDREDUCEMAC
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using namespace ondrix::ondsp;

namespace {

bool isRankOneMemRefOf(Type type, Type element) {
  auto memref = llvm::dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType() == element;
}

struct ScalarizeOndspFixedReduceMac final
    : ondrix::impl::ScalarizeOndspFixedReduceMacBase<ScalarizeOndspFixedReduceMac> {
  void runOnOperation() override {
    SmallVector<ReduceMacOp> candidates;
    getOperation()->walk([&](ReduceMacOp op) { candidates.push_back(op); });
    for (ReduceMacOp reduce : candidates) {
      auto accumulator = llvm::dyn_cast<AccType>(reduce.getInitial().getType());
      auto numeric = llvm::dyn_cast<FixedAttr>(reduce.getNumeric());
      if (!accumulator || !numeric || !reduce.getProduct() ||
          !isSingleLaneAccumulator(accumulator) ||
          !isRankOneMemRefOf(reduce.getLhs().getType(), numeric.getStorage()) ||
          !isRankOneMemRefOf(reduce.getRhs().getType(), numeric.getStorage()))
        continue;

      OpBuilder builder(reduce);
      FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
          ondrix::conversion::createRankOneMemRefReductionBounds(
              reduce, reduce.getLhs(), reduce.getRhs(), numeric.getStorage(),
              "reduce_mac scalarization", builder);
      if (failed(bounds))
        return signalPassFailure();

      Location loc = reduce.getLoc();
      Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(
          loc, bounds->lowerBound, bounds->upperBound, step, ValueRange{reduce.getInitial()},
          [&](OpBuilder &body, Location bodyLoc, Value index, ValueRange iterArgs) {
            Value sample = body.create<memref::LoadOp>(bodyLoc, reduce.getLhs(), index);
            Value coefficient = body.create<memref::LoadOp>(bodyLoc, reduce.getRhs(), index);
            Value next = body.create<MacOp>(bodyLoc, iterArgs.front().getType(), iterArgs.front(),
                                            sample, coefficient, numeric, *reduce.getProduct());
            body.create<scf::YieldOp>(bodyLoc, next);
          });
      reduce.getResult().replaceAllUsesWith(loop.getResult(0));
      reduce.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createScalarizeOndspFixedReduceMacPass() {
  return std::make_unique<ScalarizeOndspFixedReduceMac>();
}
