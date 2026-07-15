#ifndef ONDRIX_SUPPORT_DSPTYPEUTILS_H
#define ONDRIX_SUPPORT_DSPTYPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace ondrix {

inline bool isScalableVectorType(mlir::Type type) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return vector.isScalable();
  return false;
}

inline bool containsScalableVectorType(mlir::Type type) {
  return type
      .walk([](mlir::VectorType vector) {
        return vector.isScalable() ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

// Buffer reads and runtime-shaped values may gain observable checks during
// lowering, so operations using them require conservative speculation.
inline bool requiresConservativeDSPSpeculation(mlir::Type type) {
  if (mlir::isa<mlir::BaseMemRefType>(type) || isScalableVectorType(type))
    return true;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return !shaped.hasRank() || !shaped.hasStaticShape();
  return false;
}

} // namespace ondrix

#endif
