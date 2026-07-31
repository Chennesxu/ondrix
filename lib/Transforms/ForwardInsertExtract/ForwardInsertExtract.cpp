#include "ondrix/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_FORWARDONDRIXINSERTEXTRACT
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// Compile-time index vector of a tensor read or write, or nothing when any
// index is not a constant. A non-constant index cannot be compared for
// equality, so it terminates the walk instead of being skipped over.
std::optional<llvm::SmallVector<int64_t>> constantIndices(ValueRange indices) {
  llvm::SmallVector<int64_t> constants;
  constants.reserve(indices.size());
  for (Value index : indices) {
    std::optional<int64_t> constant = getConstantIntValue(index);
    if (!constant)
      return std::nullopt;
    constants.push_back(*constant);
  }
  return constants;
}

// Scalar written at `readIndices` by the nearest enclosing insert chain, or a
// null value when the chain does not prove which scalar the read observes.
Value forwardedScalar(Value source, ArrayRef<int64_t> readIndices) {
  while (auto insert = source.getDefiningOp<tensor::InsertOp>()) {
    std::optional<llvm::SmallVector<int64_t>> writeIndices = constantIndices(insert.getIndices());
    // A write with an unknown index may be the one the read observes and
    // cannot be walked past. The verifier ties every chained insert to the
    // extract tensor's rank, so the index counts always agree.
    if (!writeIndices)
      return Value();
    assert(writeIndices->size() == readIndices.size() && "insert chain changed rank");
    if (llvm::ArrayRef<int64_t>(*writeIndices) == readIndices)
      return insert.getScalar();
    source = insert.getDest();
  }
  // The chain ended at some other producer — a block argument, a
  // `tensor.empty`, or any op this pass does not model — without a matching
  // write. The read stays as it is.
  return Value();
}

class ForwardOndrixInsertExtractPass final
    : public ondrix::impl::ForwardOndrixInsertExtractBase<ForwardOndrixInsertExtractPass> {
public:
  using ondrix::impl::ForwardOndrixInsertExtractBase<
      ForwardOndrixInsertExtractPass>::ForwardOndrixInsertExtractBase;

  void runOnOperation() override {
    // One sweep in walk order. A read whose index is produced by a
    // not-yet-forwarded extract in a later block can be missed; the miss
    // leaves the read in its declared safe unforwarded state, never on a
    // wrong value. Collect first because forwarding erases the visited read.
    llvm::SmallVector<tensor::ExtractOp> reads;
    getOperation().walk([&](tensor::ExtractOp op) { reads.push_back(op); });
    for (tensor::ExtractOp read : reads) {
      std::optional<llvm::SmallVector<int64_t>> readIndices = constantIndices(read.getIndices());
      if (!readIndices)
        continue;
      Value scalar = forwardedScalar(read.getTensor(), *readIndices);
      if (!scalar)
        continue;
      // The read and the written scalar are the same SSA value, so the
      // replacement preserves every numeric boundary unchanged.
      read.getResult().replaceAllUsesWith(scalar);
      read.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createForwardOndrixInsertExtractPass() {
  return std::make_unique<ForwardOndrixInsertExtractPass>();
}
