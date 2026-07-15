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

inline bool isDynamicOrUnrankedShapedType(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return !shaped.hasRank() || !shaped.hasStaticShape();
  return false;
}

inline bool containsDynamicOrUnrankedShapedType(mlir::Type type) {
  return type
      .walk([](mlir::Type nested) {
        return isDynamicOrUnrankedShapedType(nested) ? mlir::WalkResult::interrupt()
                                                     : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

inline mlir::Type getElementTypeOrSelf(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return shaped.getElementType();
  return type;
}

inline bool isDSPScalarValueType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

// Elementwise values must remain in the same scalar/vector/tensor container
// and preserve every static dimension. Element types may differ for numeric
// conversions.
inline bool haveSameElementwiseShape(mlir::Type lhs, mlir::Type rhs) {
  auto lhsShaped = mlir::dyn_cast<mlir::ShapedType>(lhs);
  auto rhsShaped = mlir::dyn_cast<mlir::ShapedType>(rhs);
  if (static_cast<bool>(lhsShaped) != static_cast<bool>(rhsShaped))
    return false;
  if (!lhsShaped)
    return isDSPScalarValueType(lhs) && isDSPScalarValueType(rhs);
  if (mlir::isa<mlir::VectorType>(lhs) != mlir::isa<mlir::VectorType>(rhs) ||
      mlir::isa<mlir::TensorType>(lhs) != mlir::isa<mlir::TensorType>(rhs) ||
      mlir::isa<mlir::BaseMemRefType>(lhs) != mlir::isa<mlir::BaseMemRefType>(rhs))
    return false;
  return lhsShaped.getShape() == rhsShaped.getShape();
}

// Buffer reads and runtime-shaped values may gain observable checks during
// lowering, so operations using them require conservative speculation.
inline bool requiresConservativeDSPSpeculation(mlir::Type type) {
  if (mlir::isa<mlir::BaseMemRefType>(type) || isScalableVectorType(type))
    return true;
  return isDynamicOrUnrankedShapedType(type);
}

} // namespace ondrix

#endif
