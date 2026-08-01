#ifndef ONDRIX_CONVERSION_UTILS_MEMREFLAYOUTUTILS_H
#define ONDRIX_CONVERSION_UTILS_MEMREFLAYOUTUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace ondrix::conversion {

/// Returns whether a memref lives in the memory space the Vector to LLVM
/// lowering accepts. MLIR 17 rejects nonzero address spaces there, so every
/// consumer that plans a `vector.load` or `vector.store` shares this gate
/// instead of restating it.
bool hasDefaultLLVMVectorMemorySpace(mlir::MemRefType type);

/// Walks a memref value back to the storage it views. Casts, subviews, and the
/// other view-like operations change how an allocation is addressed but never
/// which allocation it is, so two values reaching the same base address the
/// same storage.
mlir::Value resolveMemRefBase(mlir::Value value);

/// Returns whether two memref values may address the same storage.
///
/// This decides only what is statically decidable: values that reach the same
/// base, and distinct reads of the same global. Two distinct FUNCTION ENTRY
/// arguments that happen to alias at run time are outside what any local rule
/// can see, so callers that rewrite the order of loads against stores must
/// state that residual precondition rather than assume this answered it. A
/// non-entry block argument is not part of that residual — its incoming
/// operands are in-function facts no caller can change — so it is refused as
/// opaque rather than deferred to a precondition.
bool mayShareStorage(mlir::Value lhs, mlir::Value rhs);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_MEMREFLAYOUTUTILS_H
