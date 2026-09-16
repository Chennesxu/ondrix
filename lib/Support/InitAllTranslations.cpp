#include "ondrix/InitAllTranslations.h"

#include "ondrix/InitAllDialects.h"
#include "ondrix/Transforms/CEntryPoints.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"

void ondrix::registerAllOndrixTranslations() {
  static mlir::TranslateFromMLIRRegistration cHeader(
      "mlir-to-ondrix-c-header",
      "Print the C header of the plain-pointer entry points an LLVM-dialect module defines",
      [](mlir::Operation *op, llvm::raw_ostream &os) -> mlir::LogicalResult {
        auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
        if (!module)
          return op->emitError("expected a module");
        return printOndrixCEntryHeader(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::LLVM::LLVMDialect>();
        registerAllOndrixDialects(registry);
      });
}
