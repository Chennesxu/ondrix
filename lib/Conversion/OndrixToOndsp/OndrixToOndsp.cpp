#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"
#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class ConvertOndrixToOndspPass
    : public PassWrapper<ConvertOndrixToOndspPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOndrixToOndspPass)

  StringRef getArgument() const final { return "convert-ondrix-to-ondsp"; }
  StringRef getDescription() const final {
    return "Lower ondrix algorithm intent ops to ondsp semantic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ondrix::ir::OndrixDialect, ondrix::ondsp::OndspDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "ondrix.fir" || name == "ondrix.dot" || name == "ondrix.butterfly" ||
          name == "ondrix.quantize")
        worklist.push_back(op);
    });

    OpBuilder builder(module.getContext());
    for (Operation *op : worklist) {
      builder.setInsertionPoint(op);
      OperationState state(op->getLoc(), getTargetName(op));
      state.addOperands(op->getOperands());
      state.addTypes(op->getResultTypes());

      for (NamedAttribute attr : op->getAttrs())
        state.addAttribute(attr.getName(), attr.getValue());

      Operation *replacement = builder.create(state);
      op->replaceAllUsesWith(replacement);
      op->erase();
    }
  }

private:
  static StringRef getTargetName(Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "ondrix.fir" || name == "ondrix.dot")
      return "ondsp.reduce_mac";
    if (name == "ondrix.butterfly")
      return "ondsp.cx_butterfly";
    return "ondsp.convert";
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndrixToOndspPass() {
  return std::make_unique<ConvertOndrixToOndspPass>();
}

void ondrix::registerConvertOndrixToOndspPass() { PassRegistration<ConvertOndrixToOndspPass>(); }

void ondrix::registerConversionPasses() {
  registerConvertOndrixToOndspPass();
  registerConvertOndspToOrtumCorePass();
}
