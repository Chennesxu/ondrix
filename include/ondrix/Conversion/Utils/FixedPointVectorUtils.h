#ifndef ONDRIX_CONVERSION_UTILS_FIXEDPOINTVECTORUTILS_H
#define ONDRIX_CONVERSION_UTILS_FIXEDPOINTVECTORUTILS_H

#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/IR/Builders.h"

namespace ondrix::conversion {

/// Vector product terms and their target-independent fixed-point domain.
class FixedVectorProductTerms final {
public:
  mlir::Value getTerms() const { return terms; }
  ondrix::ondsp::FixedAttr getNumeric() const { return numeric; }

private:
  friend mlir::FailureOr<FixedVectorProductTerms>
  lowerFixedVectorProductTerms(mlir::Operation *anchor, ondrix::ondsp::AccType accumulator,
                               ondrix::ondsp::FixedAttr numeric, ondrix::ondsp::ProductAttr product,
                               mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder);

  FixedVectorProductTerms(mlir::Value terms, ondrix::ondsp::FixedAttr numeric)
      : terms(terms), numeric(numeric) {}

  mlir::Value terms;
  ondrix::ondsp::FixedAttr numeric;
};

/// One widened horizontal sum and its fixed-point domain.
struct FixedVectorHorizontalSum {
  mlir::Value sum;
  ondrix::ondsp::FixedAttr numeric;
};

/// Materializes exact product terms for a supported fixed-width Vector MAC
/// domain. Unsupported numeric and accumulator combinations fail closed.
mlir::FailureOr<FixedVectorProductTerms>
lowerFixedVectorProductTerms(mlir::Operation *anchor, ondrix::ondsp::AccType accumulator,
                             ondrix::ondsp::FixedAttr numeric, ondrix::ondsp::ProductAttr product,
                             mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder);

/// Validates product terms, widens them to i64, and forms their horizontal sum.
/// This helper only materializes arithmetic; callers own reassociation legality.
mlir::FailureOr<FixedVectorHorizontalSum>
lowerFixedVectorHorizontalSum(mlir::Operation *anchor, const FixedVectorProductTerms &terms,
                              mlir::OpBuilder &builder);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_FIXEDPOINTVECTORUTILS_H
