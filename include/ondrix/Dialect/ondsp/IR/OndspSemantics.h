#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPSEMANTICS_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ondrix::ondsp {

/// The two rewrites `contract = fast` declares, named so that a lowering has
/// to say which one a site is spending (semantics: FpContractMode in
/// OndspEnums.td).
enum class FastPermission {
  /// Regroup the additive tree of a reduction. Only a reduction has one: a
  /// recursion body and a bare elementwise product do not.
  ReassociateReductionTerms,
  /// Select one fused multiply-add event for a term in place of a rounded
  /// product followed by an addition.
  FuseMultiplyAdd,
};

/// Fast-math flags to attach where the COMPILER spends `permission` itself:
/// none, for either permission. The produced schedule already embodies the
/// choice, so what reaches the audit point is the selected graph and not a
/// licence to select. Delegating instead — leaving `reassoc` or `contract` on
/// the operation — hands the choice to LLVM; no lowering does, so there is no
/// entry point for it here.
///
/// Delegation is not portable: de-fusion of `llvm.fma` needs both `reassoc`
/// and a target with no fused instruction, so a delegated permission makes the
/// realized graph target dependent from one declaration. Measured, not derived
/// — `test/Target/fp_permission_fmf.ll` pins it.
inline mlir::arith::FastMathFlags consumeFastPermission(FastPermission permission) {
  (void)permission;
  return mlir::arith::FastMathFlags::none;
}

/// Target-independent raw storage and fractional position of a product.
struct ProductSemantics {
  unsigned rawWidth;
  unsigned frac;
  ProductSelection selection;
};

/// Classifies whether an ordered fixed-point reduction may be reassociated.
enum class ReductionReassociationSafety {
  MustPreserveOrder,
  ExactModulo,
};

/// Whether a transform preserves the declared numeric behavior.
enum class TransformExactness {
  Exact,
  Illegal,
};

/// Evidence used to justify a transform classification.
enum class TransformJustification {
  None,
  AlgebraicIdentity,
  FixedWidthModulo,
};

/// Separates transform exactness from the evidence establishing it.
class TransformLegality {
public:
  static TransformLegality getAlgebraicIdentity();
  static TransformLegality getFixedWidthModulo();
  static TransformLegality getIllegal();

  bool isLegal() const { return exactness != TransformExactness::Illegal; }
  bool isExact() const { return exactness == TransformExactness::Exact; }
  bool isExactWith(TransformJustification expected) const {
    return isExact() && justification == expected;
  }
  TransformExactness getExactness() const { return exactness; }
  TransformJustification getJustification() const { return justification; }

private:
  TransformLegality(TransformExactness exactness, TransformJustification justification)
      : exactness(exactness), justification(justification) {}

  TransformExactness exactness;
  TransformJustification justification;
};

/// Numeric decision for the distributive pairing identity. Product semantics
/// are derived from the actual attributes rather than accepted from a caller.
struct DistributivePairingSemantics {
  ProductSemantics product;
  TransformLegality legalityWithoutRangeProof;
  bool exactBeforeAccumulatorOverflow;
};

/// Verifies whether a fixed or floating-point policy carries the required
/// product-selection attribute.
mlir::LogicalResult verifyProductPolicy(mlir::Operation *op, mlir::Attribute numeric,
                                        std::optional<ProductAttr> product);

/// Admits only f32 on an executable floating-point path. Every such path in
/// the catalog is f32: nothing lowers, references, or measures another format,
/// so accepting one would hand a program to a schedule stage no evidence
/// covers. Widening the vocabulary is a per-operation decision with its own
/// evidence, not a consequence of `FpAttr` being format-parametric.
mlir::LogicalResult verifyExecutableFpFormat(mlir::Operation *op, FpAttr numeric,
                                             llvm::StringRef executable);

/// Storage geometry of one executable packed-complex butterfly profile. The
/// two profiles differ in exactly one number, so every width-dependent rule
/// below is derived from `storageWidth` rather than restated per profile.
struct PackedComplexProfile {
  /// Signed fixed-point storage width of one component; frac is width - 1.
  unsigned storageWidth;
  /// Packed container width, always two components side by side.
  unsigned containerWidth;
};

/// Returns the executable packed-complex profile a layout denotes, or
/// std::nullopt for a layout that has no executable butterfly contract. Split,
/// interleaved, and the real-high packed spelling deliberately have none.
std::optional<PackedComplexProfile> getPackedComplexProfile(ComplexLayout layout);

/// Verifies the executable packed radix-2 butterfly profile the layout
/// selects: signed Q(storageWidth-1) numeric in matching storage, an exact
/// full product, one product requantization by storageWidth - 1, and one
/// output scale by 1, both nearest-even and saturating to the component
/// storage. Fails closed on any layout without an executable profile.
mlir::LogicalResult verifyPackedButterflyPolicy(mlir::Operation *op, CxLayoutAttr layout,
                                                mlir::Attribute numeric, ProductAttr product,
                                                ScaleAttr productScale, ScaleAttr outputScale);

/// Returns the raw product width, fractional position, and exact bit selection
/// without applying target-specific arithmetic behavior.
mlir::FailureOr<ProductSemantics> inferProductSemantics(mlir::Operation *op, FixedAttr numeric,
                                                        ProductAttr product);

/// Returns the target-independent reassociation rule implied solely by the
/// accumulator update policy.
ReductionReassociationSafety classifyReductionReassociation(OverflowMode updateOverflow);

/// Returns whether removing a multiplication by zero preserves the numeric
/// behavior. Floating-point zero products are not removable because NaN,
/// infinity, and signed zero remain observable.
TransformLegality classifyZeroProductElimination(mlir::Attribute numeric);

/// Classifies `h*x + h*y -> h*(x+y)` after coefficient equality has been
/// proven. The caller remains responsible for using widened intermediates.
mlir::FailureOr<DistributivePairingSemantics>
classifyDistributiveProductPairing(mlir::Operation *op, FixedAttr numeric, ProductAttr product,
                                   AccType accumulator);

/// Collects every type an operation exposes to a consumer that must classify
/// accumulators: operand, result, function signature, and block argument types.
/// Both target conversions gather exactly this set, so they share it.
void appendAccumulatorCandidateTypes(mlir::Operation *op, llvm::SmallVectorImpl<mlir::Type> &types);

/// Returns the first accumulator nested anywhere in `type` that `accepted`
/// rejects, or a null accumulator when every nested one is accepted. Consumers
/// pass their own capability predicate instead of restating the nested walk,
/// which is what keeps one conversion's notion of "supported" from drifting
/// into another's.
AccType findRejectedAccumulator(mlir::Type type, llvm::function_ref<bool(AccType)> accepted);

/// Returns whether the accumulator declares exactly one lane. Multi-lane
/// accumulators exist only for order-preserving cross-output batching and are
/// accepted by `acc_zero`, `mac`, and `acc_export` alone. Every other
/// accumulator consumer calls this and fails closed, so adding the lane
/// parameter cannot silently widen an existing presence-only check.
bool isSingleLaneAccumulator(AccType accumulator);

/// Returns whether a policy denotes signed Q15 in signless i16 storage.
bool isSignedQ15(FixedAttr numeric);

/// Returns whether a policy denotes signed Q31 in signless i32 storage.
bool isSignedQ31(FixedAttr numeric);

/// Returns whether a type denotes a signed i40/frac=30 accumulator domain.
bool isSignedI40Frac30Accumulator(AccType accumulator);

/// Returns whether a type denotes a signed i64/frac=62 accumulator domain.
bool isSignedI64Frac62Accumulator(AccType accumulator);

/// Returns whether a product policy selects the exact full product.
bool isFullProduct(ProductAttr product);

/// Returns whether a product policy selects the signed raw high half.
bool isRawHighProduct(ProductAttr product);

} // namespace ondrix::ondsp

#endif
