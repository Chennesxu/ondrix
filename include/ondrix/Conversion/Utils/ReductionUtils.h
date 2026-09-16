#ifndef ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H
#define ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

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

/// The `length` coefficients a rank-1 reduction operand reads, in read order,
/// when it is a constant global of exactly that length or a static reversed
/// view of one; anything else, or more than `maxElements`, is not resolved.
std::optional<llvm::SmallVector<llvm::APInt>>
getConstantCoefficientsInReadOrder(mlir::Value coefficients, int64_t length, int64_t maxElements);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_REDUCTIONUTILS_H
