#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace ondrix::ondsp {

/// Verifies whether a fixed or floating-point policy carries the required
/// product-selection attribute.
mlir::LogicalResult verifyProductPolicy(mlir::Operation *op, mlir::Attribute numeric,
                                        std::optional<ProductAttr> product);

/// Returns the fractional-bit position produced by the selected fixed-point
/// product without applying target-specific arithmetic behavior.
mlir::FailureOr<unsigned> inferProductFractionalBits(mlir::Operation *op, FixedAttr numeric,
                                                     ProductAttr product);

} // namespace ondrix::ondsp

#endif
