#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_APPLYONDRIXLLVMARGUMENTATTRIBUTES
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

class ApplyOndrixLlvmArgumentAttributesPass final
    : public ondrix::impl::ApplyOndrixLlvmArgumentAttributesBase<
          ApplyOndrixLlvmArgumentAttributesPass> {
public:
  void runOnOperation() override {
    StringRef marker = ondrix::kNoAliasPointerArgumentsAttr;
    getOperation().walk([&](LLVM::LLVMFuncOp function) {
      auto positions = function->getAttrOfType<DenseI64ArrayAttr>(marker);
      if (!positions)
        return;
      for (int64_t position : positions.asArrayRef()) {
        if (position < 0 || position >= static_cast<int64_t>(function.getNumArguments()) ||
            !isa<LLVM::LLVMPointerType>(function.getArgument(position).getType()))
          continue;
        function.setArgAttr(position, LLVM::LLVMDialect::getNoAliasAttrName(),
                            UnitAttr::get(&getContext()));
      }
      function->removeAttr(marker);
    });
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createApplyOndrixLlvmArgumentAttributesPass() {
  return std::make_unique<ApplyOndrixLlvmArgumentAttributesPass>();
}
