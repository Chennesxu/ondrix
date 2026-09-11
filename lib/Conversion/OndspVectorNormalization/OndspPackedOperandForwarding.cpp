#include "ondrix/Conversion/Passes.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_FORWARDONDSPPACKEDREDUCTIONOPERANDS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// The transposed scratch copy the matrix-product bufferization packs, with
/// the loop nest that fills it and the row views the ordered reductions read.
struct PackedCopy {
  memref::AllocOp alloc;
  Value source;
  scf::ForOp packNest;
  memref::DeallocOp dealloc;
  SmallVector<memref::SubViewOp> rows;
};

static bool isStaticRank2(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 2 && memref.hasStaticShape();
}

/// The pack nest: two counted loops whose bodies hold exactly one
/// `load source[k, c]; store packed[c, k]` pair over the two induction values.
static FailureOr<Value> matchPackNest(scf::ForOp outer, Value packed) {
  Block &outerBody = *outer.getBody();
  if (outerBody.getOperations().size() != 2)
    return failure();
  auto inner = dyn_cast<scf::ForOp>(outerBody.front());
  if (!inner || inner.getNumResults() != 0 || outer.getNumResults() != 0)
    return failure();
  Block &innerBody = *inner.getBody();
  if (innerBody.getOperations().size() != 3)
    return failure();
  auto load = dyn_cast<memref::LoadOp>(innerBody.front());
  auto store = dyn_cast<memref::StoreOp>(*std::next(innerBody.begin()));
  if (!load || !store || store.getMemRef() != packed || store.getValue() != load.getResult() ||
      !load.getResult().hasOneUse())
    return failure();
  Value column = outer.getInductionVar(), index = inner.getInductionVar();
  if (store.getIndices() != ValueRange{column, index} ||
      load.getIndices() != ValueRange{index, column} || !isStaticRank2(load.getMemRefType()))
    return failure();
  auto sourceType = load.getMemRefType();
  auto packedType = cast<MemRefType>(packed.getType());
  if (sourceType.getShape()[0] != packedType.getShape()[1] ||
      sourceType.getShape()[1] != packedType.getShape()[0])
    return failure();
  return load.getMemRef();
}

/// `memref.subview packed[c, 0] [1, K] [1, 1]`, the unit-stride row view the
/// bufferization hands to each reduction; every user must be a reduce_mac.
static bool isReducedPackedRow(memref::SubViewOp view, Value packed) {
  auto packedType = cast<MemRefType>(packed.getType());
  SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = view.getMixedSizes();
  SmallVector<OpFoldResult> strides = view.getMixedStrides();
  if (view.getSource() != packed || offsets.size() != 2 || isConstantIntValue(offsets[0], 0) ||
      !isConstantIntValue(offsets[1], 0) || !isConstantIntValue(sizes[0], 1) ||
      !isConstantIntValue(sizes[1], packedType.getShape()[1]) ||
      !isConstantIntValue(strides[0], 1) || !isConstantIntValue(strides[1], 1))
    return false;
  return llvm::all_of(view.getResult().getUsers(),
                      [](Operation *user) { return isa<ondrix::ondsp::ReduceMacOp>(user); });
}

static FailureOr<PackedCopy> matchPackedCopy(memref::AllocOp alloc) {
  if (!isStaticRank2(alloc.getType()) || alloc.getDynamicSizes().size())
    return failure();
  PackedCopy copy{alloc, nullptr, nullptr, nullptr, {}};
  for (Operation *user : alloc.getResult().getUsers()) {
    if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      if (!isReducedPackedRow(view, alloc.getResult()))
        return failure();
      copy.rows.push_back(view);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (copy.dealloc)
        return failure();
      copy.dealloc = dealloc;
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      auto inner = dyn_cast<scf::ForOp>(store->getParentOp());
      auto outer = inner ? dyn_cast<scf::ForOp>(inner->getParentOp()) : nullptr;
      if (!outer || (copy.packNest && copy.packNest != outer))
        return failure();
      FailureOr<Value> source = matchPackNest(outer, alloc.getResult());
      if (failed(source))
        return failure();
      copy.packNest = outer;
      copy.source = *source;
    } else {
      return failure();
    }
  }
  if (!copy.packNest || copy.rows.empty())
    return failure();
  return copy;
}

/// Read the source column through a strided view instead of the packed row.
static void forwardPackedCopy(PackedCopy &copy) {
  auto sourceType = cast<MemRefType>(copy.source.getType());
  int64_t innerCount = sourceType.getShape()[0];
  for (memref::SubViewOp view : copy.rows) {
    OpBuilder builder(view);
    Location loc = view.getLoc();
    OpFoldResult column = view.getMixedOffsets()[0];
    SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0), column};
    SmallVector<OpFoldResult> sizes{builder.getIndexAttr(innerCount), builder.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides{builder.getIndexAttr(1), builder.getIndexAttr(1)};
    auto viewType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        {innerCount}, sourceType, offsets, sizes, strides));
    Value strided =
        builder.create<memref::SubViewOp>(loc, viewType, copy.source, offsets, sizes, strides);
    for (Operation *user : llvm::make_early_inc_range(view.getResult().getUsers())) {
      auto reduce = cast<ondrix::ondsp::ReduceMacOp>(user);
      OpBuilder at(reduce);
      Value lhs = reduce.getLhs() == view.getResult() ? strided : reduce.getLhs();
      Value rhs = reduce.getRhs() == view.getResult() ? strided : reduce.getRhs();
      auto replacement = at.create<ondrix::ondsp::ReduceMacOp>(
          reduce.getLoc(), reduce.getResult().getType(), reduce.getInitial(), lhs, rhs,
          reduce.getNumeric(), reduce.getProductAttr());
      reduce.getResult().replaceAllUsesWith(replacement.getResult());
      reduce.erase();
    }
    view.erase();
  }
  copy.packNest.erase();
  if (copy.dealloc)
    copy.dealloc.erase();
  copy.alloc.erase();
}

class ForwardOndspPackedReductionOperandsPass final
    : public ondrix::impl::ForwardOndspPackedReductionOperandsBase<
          ForwardOndspPackedReductionOperandsPass> {
public:
  using ondrix::impl::ForwardOndspPackedReductionOperandsBase<
      ForwardOndspPackedReductionOperandsPass>::ForwardOndspPackedReductionOperandsBase;

  void runOnOperation() override {
    SmallVector<memref::AllocOp> allocs;
    getOperation().walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
    for (memref::AllocOp alloc : allocs) {
      FailureOr<PackedCopy> copy = matchPackedCopy(alloc);
      if (succeeded(copy))
        forwardPackedCopy(*copy);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createForwardOndspPackedReductionOperandsPass() {
  return std::make_unique<ForwardOndspPackedReductionOperandsPass>();
}
