#ifndef ONDRIX_FRONTEND_OXFRONTEND_H
#define ONDRIX_FRONTEND_OXFRONTEND_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class MLIRContext;
}

namespace ondrix::frontend {

/// Compile one experimental .ox source buffer to an MLIR module. Diagnostics
/// use source line and column coordinates and are written to `diagnostics`.
mlir::OwningOpRef<mlir::ModuleOp> compileOxSource(llvm::StringRef sourceName,
                                                  llvm::StringRef source,
                                                  mlir::MLIRContext &context,
                                                  llvm::raw_ostream &diagnostics);

} // namespace ondrix::frontend

#endif
