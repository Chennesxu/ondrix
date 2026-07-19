#include "ondrix/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_LOWERRANKONEMEMREFCOPYTOSCF
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

class LowerRankOneMemRefCopyToSCFPass final
    : public ondrix::impl::LowerRankOneMemRefCopyToSCFBase<LowerRankOneMemRefCopyToSCFPass> {
public:
  using ondrix::impl::LowerRankOneMemRefCopyToSCFBase<
      LowerRankOneMemRefCopyToSCFPass>::LowerRankOneMemRefCopyToSCFBase;

  void runOnOperation() override {
    SmallVector<memref::CopyOp> copies;
    getOperation().walk([&](memref::CopyOp copy) {
      auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
      auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
      if (sourceType && targetType && sourceType.getRank() == 1 && targetType.getRank() == 1)
        copies.push_back(copy);
    });

    IRRewriter rewriter(&getContext());
    for (memref::CopyOp copy : copies) {
      rewriter.setInsertionPoint(copy);
      Location loc = copy.getLoc();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value length = rewriter.create<memref::DimOp>(loc, copy.getSource(), zero);
      auto sourceType = cast<MemRefType>(copy.getSource().getType());
      auto snapshotType = MemRefType::get({ShapedType::kDynamic}, sourceType.getElementType());
      Value snapshot = rewriter.create<memref::AllocOp>(loc, snapshotType, ValueRange{length});
      rewriter.create<scf::ForOp>(
          loc, zero, length, one, ValueRange{},
          [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange) {
            Value value = builder.create<memref::LoadOp>(bodyLoc, copy.getSource(), index);
            builder.create<memref::StoreOp>(bodyLoc, value, snapshot, index);
            builder.create<scf::YieldOp>(bodyLoc);
          });
      rewriter.create<scf::ForOp>(
          loc, zero, length, one, ValueRange{},
          [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange) {
            Value value = builder.create<memref::LoadOp>(bodyLoc, snapshot, index);
            builder.create<memref::StoreOp>(bodyLoc, value, copy.getTarget(), index);
            builder.create<scf::YieldOp>(bodyLoc);
          });
      rewriter.create<memref::DeallocOp>(loc, snapshot);
      rewriter.eraseOp(copy);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createLowerRankOneMemRefCopyToSCFPass() {
  return std::make_unique<LowerRankOneMemRefCopyToSCFPass>();
}
