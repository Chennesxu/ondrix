#ifndef ONDRIX_CONVERSION_UTILS_CONVERSIONLEGALITY_H
#define ONDRIX_CONVERSION_UTILS_CONVERSIONLEGALITY_H

#include "llvm/ADT/STLFunctionalExtras.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

namespace mlir {
class LogicalResult;
class Operation;
class TypeConverter;
} // namespace mlir

namespace ondrix::conversion {

using TypePredicate = llvm::function_ref<bool(mlir::Type)>;
using AttributePredicate = llvm::function_ref<bool(mlir::Attribute)>;

/// Returns whether a type or one of its nested types matches `predicate`.
bool containsMatchingType(mlir::Type type, TypePredicate predicate);
bool containsMatchingType(mlir::TypeRange types, TypePredicate predicate);

/// Returns whether an attribute contains a matching nested type or attribute.
bool containsMatchingType(mlir::Attribute attribute, TypePredicate predicate);
bool containsMatchingAttribute(mlir::Attribute attribute, AttributePredicate predicate);

/// Checks TypeConverter legality together with recursive source artifact
/// predicates for operation values, metadata, and region block arguments.
bool hasLegalConvertedTypesAndAttributes(mlir::Operation *op, mlir::TypeConverter &typeConverter,
                                         TypePredicate rejectedType,
                                         AttributePredicate rejectedAttribute);

/// Rejects attributed scf.while operations whose carried types require
/// conversion because LLVM 17's structural rewrite does not preserve attrs.
mlir::LogicalResult verifySCFWhileTypeConversionSafety(mlir::Operation *root,
                                                       TypePredicate convertedType);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_CONVERSIONLEGALITY_H
