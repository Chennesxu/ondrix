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

FailureOr<ProductSemantics> inferProductSemantics(Operation *op, FixedAttr numeric,
                                                  ProductAttr product) {
  auto storage = cast<IntegerType>(numeric.getStorage());
  uint64_t storageWidth = storage.getWidth();
  uint64_t frac = numeric.getFrac();
  if (frac > std::numeric_limits<uint64_t>::max() / 2)
    return op->emitOpError("product fractional position is unrepresentable");

  uint64_t productFrac = frac * 2;
  uint64_t rawWidth = storageWidth;
  ProductBitSelection selection = ProductBitSelection::HighRaw;
  if (product.getSelection() == ProductSelection::Full) {
    if (storageWidth > std::numeric_limits<unsigned>::max() / 2)
      return op->emitOpError("full product storage width is unrepresentable");
    rawWidth = storageWidth * 2;
    selection = ProductBitSelection::Full;
  } else {
    if (productFrac < storageWidth)
      return op->emitOpError("raw high product fractional position would be negative");
    productFrac -= storageWidth;
  }

  if (rawWidth > std::numeric_limits<unsigned>::max() ||
      productFrac > std::numeric_limits<unsigned>::max())
    return op->emitOpError("product fractional position is unrepresentable");
  return ProductSemantics{static_cast<unsigned>(rawWidth), static_cast<unsigned>(productFrac),
                          selection};
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

bool isSignedQ31(FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 32 && numeric.getFrac() == 31 &&
         numeric.getSignedness() == Signedness::Signed;
}

bool isSignedI40Frac30Accumulator(AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 40 &&
         accumulator.getFrac() == 30 && accumulator.getSignedness() == Signedness::Signed;
}

bool isSignedI64Frac62Accumulator(AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 64 &&
         accumulator.getFrac() == 62 && accumulator.getSignedness() == Signedness::Signed;
}

bool isFullProduct(ProductAttr product) { return product.getSelection() == ProductSelection::Full; }

bool isRawHighProduct(ProductAttr product) {
  return product.getSelection() == ProductSelection::HighRaw;
}

} // namespace ondrix::ondsp
