#include "ondrix/Conversion/Utils/FixedPointVectorUtils.h"

#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace ondrix::conversion {
namespace {

FixedVectorHorizontalSum materializeFixedVectorHorizontalSum(Operation *anchor,
                                                             const FixedVectorProductTerms &terms,
                                                             OpBuilder &builder) {
  auto termVectorType = cast<VectorType>(terms.getTerms().getType());
  auto termStorage = cast<IntegerType>(termVectorType.getElementType());

  auto wideVectorType = VectorType::get(termVectorType.getShape(), builder.getI64Type());
  Value wideTerms = terms.getTerms();
  if (termStorage.getWidth() < 64)
    wideTerms = builder.create<arith::ExtSIOp>(anchor->getLoc(), wideVectorType, terms.getTerms());
  Value sum =
      builder.create<vector::ReductionOp>(anchor->getLoc(), vector::CombiningKind::ADD, wideTerms);
  auto numeric =
      ondrix::ondsp::FixedAttr::get(builder.getContext(), terms.getNumeric().getSignedness(),
                                    builder.getI64Type(), terms.getNumeric().getFrac());
  return FixedVectorHorizontalSum{sum, numeric};
}

} // namespace

FailureOr<FixedVectorProductTerms>
lowerFixedVectorProductTerms(Operation *anchor, ondrix::ondsp::AccType accumulator,
                             ondrix::ondsp::FixedAttr numeric, ondrix::ondsp::ProductAttr product,
                             Value lhs, Value rhs, OpBuilder &builder) {
  FailureOr<SupportedFixedMacDomain> domain =
      getSupportedFixedVectorMacDomain(anchor, accumulator, numeric, product);
  auto lhsType = dyn_cast<VectorType>(lhs.getType());
  auto rhsType = dyn_cast<VectorType>(rhs.getType());
  if (failed(domain) || !lhsType || !rhsType || lhsType.isScalable() || rhsType.isScalable() ||
      lhsType != rhsType || lhsType.getElementType() != domain->operandStorage)
    return failure();

  auto fullProductVectorType = VectorType::get(lhsType.getShape(), domain->fullProductStorage);
  Value extendedLhs = builder.create<arith::ExtSIOp>(anchor->getLoc(), fullProductVectorType, lhs);
  Value extendedRhs = builder.create<arith::ExtSIOp>(anchor->getLoc(), fullProductVectorType, rhs);
  Value fullProducts = builder.create<arith::MulIOp>(anchor->getLoc(), extendedLhs, extendedRhs);

  Value terms = fullProducts;
  if (domain->product.selection == ondrix::ondsp::ProductSelection::HighRaw) {
    IntegerAttr shiftValue =
        builder.getIntegerAttr(domain->fullProductStorage, domain->operandStorage.getWidth());
    auto shiftValues = SplatElementsAttr::get(fullProductVectorType, shiftValue);
    Value shift =
        builder.create<arith::ConstantOp>(anchor->getLoc(), fullProductVectorType, shiftValues);
    Value shifted = builder.create<arith::ShRSIOp>(anchor->getLoc(), fullProducts, shift);
    auto termVectorType = VectorType::get(lhsType.getShape(), domain->termStorage);
    terms = builder.create<arith::TruncIOp>(anchor->getLoc(), termVectorType, shifted);
  }

  auto termNumeric = ondrix::ondsp::FixedAttr::get(builder.getContext(), domain->signedness,
                                                   domain->termStorage, domain->product.frac);
  return FixedVectorProductTerms(terms, termNumeric);
}

FailureOr<FixedVectorHorizontalSum>
lowerFixedVectorHorizontalSum(Operation *anchor, const FixedVectorProductTerms &terms,
                              OpBuilder &builder) {
  auto vectorType = dyn_cast<VectorType>(terms.getTerms().getType());
  if (!vectorType)
    return failure();
  auto termStorage = dyn_cast<IntegerType>(vectorType.getElementType());
  if (!termStorage || !termStorage.isSignless() || vectorType.isScalable() ||
      termStorage.getWidth() > 64 || terms.getNumeric().getStorage() != termStorage ||
      terms.getNumeric().getSignedness() != ondrix::ondsp::Signedness::Signed)
    return failure();
  return materializeFixedVectorHorizontalSum(anchor, terms, builder);
}

} // namespace ondrix::conversion
