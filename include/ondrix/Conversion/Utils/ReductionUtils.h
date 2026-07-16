#ifndef ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H
#define ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace ondrix::conversion {

struct RankOneReductionBounds {
  mlir::MemRefType lhsType;
  mlir::MemRefType rhsType;
  mlir::Value lowerBound;
  mlir::Value upperBound;
};

/// Validates a rank-1 memref reduction domain, materializes its bounds, and
/// inserts the runtime equal-length proof required for dynamic dimensions.
mlir::FailureOr<RankOneReductionBounds>
createRankOneMemRefReductionBounds(mlir::Operation *op, mlir::Value lhs, mlir::Value rhs,
                                   mlir::Type elementType, llvm::StringRef consumer,
                                   mlir::OpBuilder &builder);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H
