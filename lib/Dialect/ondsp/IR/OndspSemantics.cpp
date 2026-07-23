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
    if (failed(inferProductSemantics(op, cast<FixedAttr>(numeric), *product)))
      return failure();
    return success();
  }

  if (product)
    return op->emitOpError("floating-point numeric policy must not specify a product attribute");
  return success();
}

static LogicalResult verifyButterflyScale(Operation *op, ScaleAttr scale, unsigned rightShift,
                                          StringRef name) {
  if (scale.getPreShiftLeft() != 0 || scale.getPostShiftRight() != rightShift)
    return op->emitOpError() << name
                             << " requires pre_shift_left=0 and post_shift_right=" << rightShift;
  if (scale.getRounding() != RoundingMode::NearestEven)
    return op->emitOpError() << name << " requires nearest_even rounding";
  if (scale.getOverflow() != OverflowMode::Saturate)
    return op->emitOpError() << name << " requires saturating overflow";
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  if (!destination || !destination.isSignless() || destination.getWidth() != 16)
    return op->emitOpError() << name << " requires signless i16 destination storage";
  return success();
}

LogicalResult verifyPackedQ15ButterflyPolicy(Operation *op, Attribute numeric, ProductAttr product,
                                             ScaleAttr productScale, ScaleAttr outputScale) {
  auto fixed = dyn_cast<FixedAttr>(numeric);
  if (!fixed || !isSignedQ15(fixed))
    return op->emitOpError("packed butterfly requires signed Q15 numeric semantics");
  if (!isFullProduct(product))
    return op->emitOpError("packed butterfly requires product = #ondsp.product<full>");
  if (failed(verifyButterflyScale(op, productScale, 15, "product_scale")))
    return failure();
  return verifyButterflyScale(op, outputScale, 1, "output_scale");
}

FailureOr<ProductSemantics> inferProductSemantics(Operation *op, FixedAttr numeric,
                                                  ProductAttr product) {
  if (numeric.getSignedness() != Signedness::Signed)
    return op->emitOpError("fixed product semantics currently require a signed numeric policy");

  auto storage = cast<IntegerType>(numeric.getStorage());
  uint64_t storageWidth = storage.getWidth();
  uint64_t frac = numeric.getFrac();
  if (frac > std::numeric_limits<uint64_t>::max() / 2)
    return op->emitOpError("product fractional position is unrepresentable");

  uint64_t productFrac = frac * 2;
  switch (product.getSelection()) {
  case ProductSelection::Full:
    if (storageWidth > std::numeric_limits<unsigned>::max() / 2)
      return op->emitOpError("full product storage width is unrepresentable");
    if (storageWidth * 2 > std::numeric_limits<unsigned>::max() ||
        productFrac > std::numeric_limits<unsigned>::max())
      return op->emitOpError("product fractional position is unrepresentable");
    return ProductSemantics{static_cast<unsigned>(storageWidth * 2),
                            static_cast<unsigned>(productFrac), ProductSelection::Full};
  case ProductSelection::HighRaw:
    if (productFrac < storageWidth)
      return op->emitOpError("raw high product fractional position would be negative");
    productFrac -= storageWidth;
    if (storageWidth > std::numeric_limits<unsigned>::max() ||
        productFrac > std::numeric_limits<unsigned>::max())
      return op->emitOpError("product fractional position is unrepresentable");
    return ProductSemantics{static_cast<unsigned>(storageWidth), static_cast<unsigned>(productFrac),
                            ProductSelection::HighRaw};
  }
  return op->emitOpError("unsupported fixed product selection");
}

ReductionReassociationSafety classifyReductionReassociation(OverflowMode updateOverflow) {
  if (updateOverflow == OverflowMode::Wrap)
    return ReductionReassociationSafety::ExactModulo;
  return ReductionReassociationSafety::MustPreserveOrder;
}

TransformLegality TransformLegality::getAlgebraicIdentity() {
  return TransformLegality(TransformExactness::Exact, TransformJustification::AlgebraicIdentity);
}

TransformLegality TransformLegality::getFixedWidthModulo() {
  return TransformLegality(TransformExactness::Exact, TransformJustification::FixedWidthModulo);
}

TransformLegality TransformLegality::getIllegal() {
  return TransformLegality(TransformExactness::Illegal, TransformJustification::None);
}

TransformLegality classifyZeroProductElimination(Attribute numeric) {
  return isa<FixedAttr>(numeric) ? TransformLegality::getAlgebraicIdentity()
                                 : TransformLegality::getIllegal();
}

FailureOr<DistributivePairingSemantics> classifyDistributiveProductPairing(Operation *op,
                                                                           FixedAttr numeric,
                                                                           ProductAttr product,
                                                                           AccType accumulator) {
  FailureOr<ProductSemantics> productSemantics = inferProductSemantics(op, numeric, product);
  if (failed(productSemantics))
    return failure();

  bool exactBeforeAccumulatorOverflow = numeric.getSignedness() == Signedness::Signed &&
                                        productSemantics->selection == ProductSelection::Full &&
                                        accumulator.getSignedness() == numeric.getSignedness() &&
                                        accumulator.getFrac() == productSemantics->frac;
  TransformLegality legality = TransformLegality::getIllegal();
  if (exactBeforeAccumulatorOverflow && accumulator.getUpdateOverflow() == OverflowMode::Wrap)
    legality = TransformLegality::getFixedWidthModulo();
  return DistributivePairingSemantics{*productSemantics, legality, exactBeforeAccumulatorOverflow};
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
