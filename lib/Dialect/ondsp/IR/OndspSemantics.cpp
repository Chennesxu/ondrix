#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/IR/BuiltinTypes.h"

#include <limits>

using namespace mlir;

namespace ondrix::ondsp {

LogicalResult verifyProductPolicy(Operation *op, Attribute numeric,
                                  std::optional<ProductAttr> product) {
  if (isa<FixedAttr>(numeric)) {
    if (!product)
      return op->emitOpError("fixed numeric policy requires a product attribute");
    return success();
  }

  if (product)
    return op->emitOpError("floating-point numeric policy must not specify a product attribute");
  return success();
}

FailureOr<unsigned> inferProductFractionalBits(Operation *op, FixedAttr numeric,
                                               ProductAttr product) {
  auto storage = cast<IntegerType>(numeric.getStorage());
  uint64_t frac = numeric.getFrac();
  if (frac > std::numeric_limits<uint64_t>::max() / 2)
    return op->emitOpError("product fractional position is unrepresentable");

  uint64_t productFrac = frac * 2;
  if (product.getSelection() == ProductSelection::High) {
    if (productFrac < storage.getWidth())
      return op->emitOpError("high product fractional position would be negative");
    productFrac -= storage.getWidth();
  }

  if (productFrac > std::numeric_limits<unsigned>::max())
    return op->emitOpError("product fractional position is unrepresentable");
  return static_cast<unsigned>(productFrac);
}

ReductionReassociationSafety classifyReductionReassociation(OverflowMode updateOverflow,
                                                            ReductionRangeProof rangeProof) {
  if (rangeProof == ReductionRangeProof::AllOriginalAndReassociatedSumsFit)
    return ReductionReassociationSafety::ProvenNoOverflow;
  if (updateOverflow == OverflowMode::Wrap)
    return ReductionReassociationSafety::ExactModulo;
  return ReductionReassociationSafety::MustPreserveOrder;
}

bool isSignedQ15(FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 16 && numeric.getFrac() == 15 &&
         numeric.getSignedness() == Signedness::Signed;
}

bool isFullProduct(ProductAttr product) { return product.getSelection() == ProductSelection::Full; }

} // namespace ondrix::ondsp
