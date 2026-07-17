#ifndef ONDRIX_CONVERSION_UTILS_FIXEDPOINTDOMAINUTILS_H
#define ONDRIX_CONVERSION_UTILS_FIXEDPOINTDOMAINUTILS_H

#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/IR/BuiltinTypes.h"

namespace ondrix::conversion {

/// Describes the storage and product semantics of one implemented fixed-point
/// MAC domain.
struct SupportedFixedMacDomain {
  ondrix::ondsp::Signedness signedness;
  mlir::IntegerType operandStorage;
  mlir::IntegerType fullProductStorage;
  mlir::IntegerType termStorage;
  ondrix::ondsp::ProductSemantics product;
};

/// Returns whether the generic scalar consumer implements this exact domain.
bool isSupportedFixedScalarMacDomain(ondrix::ondsp::AccType accumulator,
                                     ondrix::ondsp::FixedAttr numeric,
                                     ondrix::ondsp::ProductAttr product);

/// Returns whether the fixed-width Vector consumers implement this exact
/// domain. This is intentionally separate from scalar legality so adding a
/// scalar-only domain cannot broaden Vector legality implicitly.
bool isSupportedFixedVectorMacDomain(ondrix::ondsp::AccType accumulator,
                                     ondrix::ondsp::FixedAttr numeric,
                                     ondrix::ondsp::ProductAttr product);

/// Materializes the storage types and target-independent product semantics for
/// a scalar-supported domain. Unsupported domains fail closed.
mlir::FailureOr<SupportedFixedMacDomain>
getSupportedFixedScalarMacDomain(mlir::Operation *op, ondrix::ondsp::AccType accumulator,
                                 ondrix::ondsp::FixedAttr numeric,
                                 ondrix::ondsp::ProductAttr product);

/// Materializes the storage types and target-independent product semantics for
/// a Vector-supported domain. Unsupported domains fail closed.
mlir::FailureOr<SupportedFixedMacDomain>
getSupportedFixedVectorMacDomain(mlir::Operation *op, ondrix::ondsp::AccType accumulator,
                                 ondrix::ondsp::FixedAttr numeric,
                                 ondrix::ondsp::ProductAttr product);

} // namespace ondrix::conversion

#endif // ONDRIX_CONVERSION_UTILS_FIXEDPOINTDOMAINUTILS_H
