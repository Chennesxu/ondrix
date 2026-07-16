#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace ondrix::ondsp {

/// Describes the proof available to a reduction transform. The stronger case
/// covers every intermediate sum in both the source order and the proposed
/// reassociation, not merely the final mathematical sum.
enum class ReductionRangeProof {
  None,
  AllOriginalAndReassociatedSumsFit,
};

/// Classifies whether an ordered fixed-point reduction may be reassociated.
enum class ReductionReassociationSafety {
  MustPreserveOrder,
  ExactModulo,
  ProvenNoOverflow,
};

/// Verifies whether a fixed or floating-point policy carries the required
/// product-selection attribute.
mlir::LogicalResult verifyProductPolicy(mlir::Operation *op, mlir::Attribute numeric,
                                        std::optional<ProductAttr> product);

/// Returns the fractional-bit position produced by the selected fixed-point
/// product without applying target-specific arithmetic behavior.
mlir::FailureOr<unsigned> inferProductFractionalBits(mlir::Operation *op, FixedAttr numeric,
                                                     ProductAttr product);

/// Returns the target-independent reassociation rule for an accumulator
/// update policy and an optional range-analysis proof.
ReductionReassociationSafety
classifyReductionReassociation(OverflowMode updateOverflow,
                               ReductionRangeProof rangeProof = ReductionRangeProof::None);

/// Returns whether a policy denotes signed Q15 in signless i16 storage.
bool isSignedQ15(FixedAttr numeric);

/// Returns whether a type denotes a signed i40/frac=30 accumulator domain.
bool isSignedI40Frac30Accumulator(AccType accumulator);

/// Returns whether a product policy selects the exact full product.
bool isFullProduct(ProductAttr product);

} // namespace ondrix::ondsp

#endif
