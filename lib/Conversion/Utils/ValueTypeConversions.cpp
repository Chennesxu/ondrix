#include "ondrix/Conversion/Utils/ValueTypeConversions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace ondrix::conversion {
namespace {

class UnrealizedCastTypeConversion final : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    // No attribute propagation: these structural ops have no inherent
    // attributes and no pass stamps discardable metadata on them, so a
    // wholesale copy could only smuggle unapproved attributes.
    auto replacement =
        rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), resultTypes, adaptor.getInputs());
    rewriter.replaceOp(op, replacement.getResults());
    return success();
  }
};

class SelectOpTypeConversion final : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    // Discardable metadata on a select survives the structural type
    // conversion; accumulator_control_flow_q15.mlir witnesses the contract.
    auto replacement =
        rewriter.create<arith::SelectOp>(op.getLoc(), resultType, adaptor.getCondition(),
                                         adaptor.getTrueValue(), adaptor.getFalseValue());
    replacement->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

} // namespace

void populateValueTypeConversionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<SelectOpTypeConversion, UnrealizedCastTypeConversion>(typeConverter,
                                                                     patterns.getContext());
}

} // namespace ondrix::conversion
