#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_DECLAREONDRIXARGUMENTSREADONLY
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

class DeclareOndrixArgumentsReadOnlyPass final
    : public ondrix::impl::DeclareOndrixArgumentsReadOnlyBase<DeclareOndrixArgumentsReadOnlyPass> {
public:
  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation().walk([&](func::FuncOp function) {
      for (BlockArgument argument : function.getArguments())
        if (isa<TensorType>(argument.getType()))
          function.setArgAttr(argument.getArgNumber(),
                              bufferization::BufferizationDialect::kWritableAttrName,
                              builder.getBoolAttr(false));
    });
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createDeclareOndrixArgumentsReadOnlyPass() {
  return std::make_unique<DeclareOndrixArgumentsReadOnlyPass>();
}
