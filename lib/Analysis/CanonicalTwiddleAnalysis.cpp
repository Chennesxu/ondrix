#include "ondrix/Analysis/CanonicalTwiddleAnalysis.h"

#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace ondrix::analysis {

namespace {

bool hasSupportedScaleShape(ondsp::ScaleAttr scale, unsigned rightShift) {
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  return scale.getPreShiftLeft() == 0 && scale.getPostShiftRight() == rightShift && destination &&
         destination.isSignless() && destination.getWidth() == 16;
}

} // namespace

StringRef stringifyCanonicalPackedQ15TwiddleStatus(CanonicalPackedQ15TwiddleStatus status) {
  switch (status) {
  case CanonicalPackedQ15TwiddleStatus::Authorized:
    return "authorized";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedValueDomain:
    return "unsupported_value_domain";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedLayout:
    return "unsupported_layout";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedNumeric:
    return "unsupported_numeric";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedProduct:
    return "unsupported_product";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedScale:
    return "unsupported_scale";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedRounding:
    return "unsupported_rounding";
  case CanonicalPackedQ15TwiddleStatus::UnsupportedOverflow:
    return "unsupported_overflow";
  case CanonicalPackedQ15TwiddleStatus::NonConstantTwiddle:
    return "nonconstant_twiddle";
  case CanonicalPackedQ15TwiddleStatus::NonCanonicalTwiddle:
    return "noncanonical_twiddle";
  }
  llvm_unreachable("unknown canonical packed-Q15 twiddle status");
}

CanonicalPackedQ15TwiddleClassification
classifyCanonicalPackedQ15Twiddle(ondsp::CxButterflyOp butterfly) {
  if (!butterfly.getTwiddle().getType().isSignlessInteger(32))
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedValueDomain, std::nullopt};
  if (butterfly.getLayout().getLayout() != ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedLayout, std::nullopt};

  auto numeric = dyn_cast<ondsp::FixedAttr>(butterfly.getNumeric());
  if (!numeric || !ondsp::isSignedQ15(numeric))
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedNumeric, std::nullopt};
  if (!ondsp::isFullProduct(butterfly.getProduct()))
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedProduct, std::nullopt};

  ondsp::ScaleAttr productScale = butterfly.getProductScale();
  ondsp::ScaleAttr outputScale = butterfly.getOutputScale();
  if (!hasSupportedScaleShape(productScale, 15) || !hasSupportedScaleShape(outputScale, 1))
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedScale, std::nullopt};
  if (productScale.getRounding() != ondsp::RoundingMode::NearestEven ||
      outputScale.getRounding() != ondsp::RoundingMode::NearestEven)
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedRounding, std::nullopt};
  if (productScale.getOverflow() != ondsp::OverflowMode::Saturate ||
      outputScale.getOverflow() != ondsp::OverflowMode::Saturate)
    return {CanonicalPackedQ15TwiddleStatus::UnsupportedOverflow, std::nullopt};

  auto constant = butterfly.getTwiddle().getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  auto integerType = integer ? dyn_cast<IntegerType>(integer.getType()) : IntegerType();
  if (!integerType || integerType.getWidth() != 32)
    return {CanonicalPackedQ15TwiddleStatus::NonConstantTwiddle, std::nullopt};

  uint64_t bits = integer.getValue().getZExtValue();
  if (bits == 0x00007fffU)
    return {CanonicalPackedQ15TwiddleStatus::Authorized, CanonicalPackedQ15TwiddleIdentity::One};
  if (bits == 0x80000000U)
    return {CanonicalPackedQ15TwiddleStatus::Authorized, CanonicalPackedQ15TwiddleIdentity::MinusJ};
  return {CanonicalPackedQ15TwiddleStatus::NonCanonicalTwiddle, std::nullopt};
}

CanonicalPackedQ15TwiddlePlan::CanonicalPackedQ15TwiddlePlan(
    ondsp::CxButterflyOp butterfly, CanonicalPackedQ15TwiddleIdentity identity)
    : subject(butterfly.getOperation()), twiddle(butterfly.getTwiddle()),
      layout(butterfly.getLayout()), numeric(butterfly.getNumeric()),
      product(butterfly.getProduct()), productScale(butterfly.getProductScale()),
      outputScale(butterfly.getOutputScale()), identity(identity) {}

CanonicalPackedQ15TwiddlePlan::CanonicalPackedQ15TwiddlePlan(CanonicalPackedQ15TwiddlePlan &&other)
    : subject(other.subject), twiddle(other.twiddle), layout(other.layout), numeric(other.numeric),
      product(other.product), productScale(other.productScale), outputScale(other.outputScale),
      identity(other.identity), consumed(other.consumed) {
  other.subject = nullptr;
  other.twiddle = {};
  other.consumed = true;
}

CanonicalPackedQ15TwiddlePlan &
CanonicalPackedQ15TwiddlePlan::operator=(CanonicalPackedQ15TwiddlePlan &&other) {
  if (this == &other)
    return *this;
  subject = other.subject;
  twiddle = other.twiddle;
  layout = other.layout;
  numeric = other.numeric;
  product = other.product;
  productScale = other.productScale;
  outputScale = other.outputScale;
  identity = other.identity;
  consumed = other.consumed;
  other.subject = nullptr;
  other.twiddle = {};
  other.consumed = true;
  return *this;
}

LogicalResult
CanonicalPackedQ15TwiddlePlan::consumeIfValid(ondsp::CxButterflyOp butterfly,
                                              CanonicalPackedQ15TwiddleConsumer consumer) && {
  if (consumed)
    return failure();
  consumed = true;
  CanonicalPackedQ15TwiddleClassification current = classifyCanonicalPackedQ15Twiddle(butterfly);
  if (subject != butterfly.getOperation() || twiddle != butterfly.getTwiddle() ||
      layout != butterfly.getLayout() || numeric != butterfly.getNumeric() ||
      product != butterfly.getProduct() || productScale != butterfly.getProductScale() ||
      outputScale != butterfly.getOutputScale() ||
      current.status != CanonicalPackedQ15TwiddleStatus::Authorized || current.identity != identity)
    return failure();
  return consumer(identity);
}

std::optional<CanonicalPackedQ15TwiddlePlan>
planCanonicalPackedQ15Twiddle(ondsp::CxButterflyOp butterfly) {
  CanonicalPackedQ15TwiddleClassification classification =
      classifyCanonicalPackedQ15Twiddle(butterfly);
  if (classification.status != CanonicalPackedQ15TwiddleStatus::Authorized ||
      !classification.identity)
    return std::nullopt;
  return CanonicalPackedQ15TwiddlePlan(butterfly, *classification.identity);
}

} // namespace ondrix::analysis
