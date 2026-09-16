#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDRIXSTATICRESULTSTOOUTPARAMS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

bool hasOnlyStaticMemRefResults(func::FuncOp *function) {
  return llvm::all_of(function->getResultTypes(), [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    return !memref || memref.hasStaticShape();
  });
}

class ConvertOndrixStaticResultsToOutParamsPass final
    : public ondrix::impl::ConvertOndrixStaticResultsToOutParamsBase<
          ConvertOndrixStaticResultsToOutParamsPass> {
public:
  void runOnOperation() override {
    bufferization::BufferResultsToOutParamsOptions options;
    options.filterFn = hasOnlyStaticMemRefResults;
    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation(), options)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixStaticResultsToOutParamsPass() {
  return std::make_unique<ConvertOndrixStaticResultsToOutParamsPass>();
}
