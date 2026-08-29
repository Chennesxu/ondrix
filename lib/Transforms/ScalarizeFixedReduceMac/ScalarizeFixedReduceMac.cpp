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
      Location loc = reduce.getLoc();

      // A short static reduction unrolls to a straight-line chain: the loop
      // form pays an index update and a branch per term, which outweighs the
      // term itself on the MAC capability (measured; 64 is the batcher's
      // unroll bound).
      auto lhsType = cast<MemRefType>(reduce.getLhs().getType());
      auto rhsType = cast<MemRefType>(reduce.getRhs().getType());
      if (!lhsType.isDynamicDim(0) && !rhsType.isDynamicDim(0) &&
          lhsType.getDimSize(0) == rhsType.getDimSize(0) && lhsType.getDimSize(0) <= 64) {
        Value chain = reduce.getInitial();
        for (int64_t term = 0; term < lhsType.getDimSize(0); ++term) {
          Value index = builder.create<arith::ConstantIndexOp>(loc, term);
          Value sample = builder.create<memref::LoadOp>(loc, reduce.getLhs(), index);
          Value coefficient = builder.create<memref::LoadOp>(loc, reduce.getRhs(), index);
          chain = builder.create<MacOp>(loc, chain.getType(), chain, sample, coefficient, numeric,
                                        *reduce.getProduct());
        }
        reduce.getResult().replaceAllUsesWith(chain);
        reduce.erase();
        continue;
      }

      FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
          ondrix::conversion::createRankOneMemRefReductionBounds(
              reduce, reduce.getLhs(), reduce.getRhs(), numeric.getStorage(),
              "reduce_mac scalarization", builder);
      if (failed(bounds))
        return signalPassFailure();

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
