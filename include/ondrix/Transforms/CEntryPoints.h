#ifndef ONDRIX_TRANSFORMS_CENTRYPOINTS_H
#define ONDRIX_TRANSFORMS_CENTRYPOINTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace ondrix {

/// Function attribute, a dictionary `{signature, names, groups}`: the
/// bufferized signature a C entry point is built from, one name per
/// parameter, and per parameter the extent group whose one length it shares
/// (-1 for a static extent). `declare-ondrix-c-entry-points` writes it on the
/// kernel; `emit-ondrix-c-entry-points` moves it to the wrapper it builds.
constexpr llvm::StringLiteral kCEntryAttr = "ondrix.c_entry";

/// The symbol every generated C entry point is named under.
constexpr llvm::StringLiteral kCEntryPrefix = "ondrix_";

/// Prints a C header declaring every entry point `emit-ondrix-c-entry-points`
/// built in `module`: element types, the length type in the target's index
/// width, and the read-only buffer contract.
mlir::LogicalResult printOndrixCEntryHeader(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace ondrix

#endif // ONDRIX_TRANSFORMS_CENTRYPOINTS_H
