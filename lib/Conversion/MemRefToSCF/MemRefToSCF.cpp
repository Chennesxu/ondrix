#include "ondrix/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_LOWERMEMREFCOPYTOSCF
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static bool isStaticCopy(memref::CopyOp copy) {
  auto sourceType = cast<MemRefType>(copy.getSource().getType());
  auto targetType = cast<MemRefType>(copy.getTarget().getType());
  return sourceType.hasStaticShape() && targetType.hasStaticShape() &&
         sourceType.getShape() == targetType.getShape();
}

static bool isLowerableCopy(memref::CopyOp copy) {
  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!sourceType || !targetType)
    return false;
  return (sourceType.getRank() == 1 && targetType.getRank() == 1) || isStaticCopy(copy);
}

// One loop per dimension of a static shape; the body sees the whole index.
static void forEachIndex(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
                         SmallVectorImpl<Value> &indices,
                         llvm::function_ref<void(OpBuilder &, Location, ValueRange)> body) {
  if (indices.size() == shape.size()) {
    body(builder, loc, indices);
    return;
  }
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value extent = builder.create<arith::ConstantIndexOp>(loc, shape[indices.size()]);
  builder.create<scf::ForOp>(loc, zero, extent, one, ValueRange{},
                             [&](OpBuilder &nested, Location nestedLoc, Value index, ValueRange) {
                               indices.push_back(index);
                               forEachIndex(nested, nestedLoc, shape, indices, body);
                               indices.pop_back();
                               nested.create<scf::YieldOp>(nestedLoc);
                             });
}

static void lowerStaticCopy(IRRewriter &rewriter, memref::CopyOp copy) {
  Location loc = copy.getLoc();
  auto sourceType = cast<MemRefType>(copy.getSource().getType());
  auto snapshotType = MemRefType::get(sourceType.getShape(), sourceType.getElementType());
  // The snapshot lives on the stack at the function entry: a static shape
  // needs no heap, and an alloca beside a copy inside a loop would grow the
  // frame on every trip.
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto function = copy->getParentOfType<func::FuncOp>())
    rewriter.setInsertionPointToStart(&function.getBody().front());
  Value snapshot = rewriter.create<memref::AllocaOp>(loc, snapshotType);
  rewriter.setInsertionPoint(copy);
  SmallVector<Value> indices;
  forEachIndex(rewriter, loc, sourceType.getShape(), indices,
               [&](OpBuilder &builder, Location bodyLoc, ValueRange index) {
                 Value value = builder.create<memref::LoadOp>(bodyLoc, copy.getSource(), index);
                 builder.create<memref::StoreOp>(bodyLoc, value, snapshot, index);
               });
  forEachIndex(rewriter, loc, sourceType.getShape(), indices,
               [&](OpBuilder &builder, Location bodyLoc, ValueRange index) {
                 Value value = builder.create<memref::LoadOp>(bodyLoc, snapshot, index);
                 builder.create<memref::StoreOp>(bodyLoc, value, copy.getTarget(), index);
               });
}

static void lowerDynamicRankOneCopy(IRRewriter &rewriter, memref::CopyOp copy) {
  Location loc = copy.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value length = rewriter.create<memref::DimOp>(loc, copy.getSource(), zero);
  auto sourceType = cast<MemRefType>(copy.getSource().getType());
  auto snapshotType = MemRefType::get({ShapedType::kDynamic}, sourceType.getElementType());
  Value snapshot = rewriter.create<memref::AllocOp>(loc, snapshotType, ValueRange{length});
  rewriter.create<scf::ForOp>(loc, zero, length, one, ValueRange{},
                              [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange) {
                                Value value = builder.create<memref::LoadOp>(
                                    bodyLoc, copy.getSource(), index);
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
}

class LowerMemRefCopyToSCFPass final
    : public ondrix::impl::LowerMemRefCopyToSCFBase<LowerMemRefCopyToSCFPass> {
public:
  using ondrix::impl::LowerMemRefCopyToSCFBase<LowerMemRefCopyToSCFPass>::LowerMemRefCopyToSCFBase;

  void runOnOperation() override {
    SmallVector<memref::CopyOp> copies;
    getOperation().walk([&](memref::CopyOp copy) {
      if (isLowerableCopy(copy))
        copies.push_back(copy);
    });
    IRRewriter rewriter(&getContext());
    for (memref::CopyOp copy : copies) {
      rewriter.setInsertionPoint(copy);
      if (isStaticCopy(copy))
        lowerStaticCopy(rewriter, copy);
      else
        lowerDynamicRankOneCopy(rewriter, copy);
      rewriter.eraseOp(copy);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createLowerMemRefCopyToSCFPass() {
  return std::make_unique<LowerMemRefCopyToSCFPass>();
}
