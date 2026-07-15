#include "ondrix/Conversion/OndspToScalar/OndspToScalar.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPQ15TOSCALAR
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static bool isSupportedQ15Accumulator(ondrix::ondsp::AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 40 &&
         accumulator.getFrac() == 30 &&
         accumulator.getSignedness() == ondrix::ondsp::Signedness::Signed;
}

class OndspQ15ToScalarTypeConverter final : public TypeConverter {
public:
  OndspQ15ToScalarTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](ondrix::ondsp::AccType type,
                     SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (!isSupportedQ15Accumulator(type))
        return failure();
      results.push_back(type.getStorage());
      return success();
    });
  }
};

static bool containsOndspAccumulator(Type type) {
  return type.walk([](ondrix::ondsp::AccType) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool containsOndspAccumulator(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsOndspAccumulator(type); });
}

static bool containsOndspAccumulator(Attribute attribute) {
  return attribute.walk([](ondrix::ondsp::AccType) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  if (!typeConverter.isLegal(op) || containsOndspAccumulator(op->getOperandTypes()) ||
      containsOndspAccumulator(op->getResultTypes()))
    return false;
  if (llvm::any_of(op->getAttrs(), [](NamedAttribute namedAttribute) {
        return containsOndspAccumulator(namedAttribute.getValue());
      }))
    return false;
  for (Region &region : op->getRegions()) {
    if (!typeConverter.isLegal(&region))
      return false;
    for (Block &block : region)
      if (containsOndspAccumulator(block.getArgumentTypes()))
        return false;
  }
  return true;
}

class AccZeroOpLowering final : public OpConversionPattern<ondrix::ondsp::AccZeroOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccZeroOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getAcc().getType());
    auto integerType = dyn_cast_or_null<IntegerType>(resultType);
    if (!integerType)
      return op.emitOpError("Q15 scalar lowering requires a supported accumulator type");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, integerType,
                                                   rewriter.getIntegerAttr(integerType, 0));
    return success();
  }
};

class ConvertOndspQ15ToScalarPass final
    : public ondrix::impl::ConvertOndspQ15ToScalarBase<ConvertOndspQ15ToScalarPass> {
public:
  using ondrix::impl::ConvertOndspQ15ToScalarBase<
      ConvertOndspQ15ToScalarPass>::ConvertOndspQ15ToScalarBase;

  void runOnOperation() override {
    OndspQ15ToScalarTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    patterns.add<AccZeroOpLowering>(typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect, cf::ControlFlowDialect>();
    target.addIllegalDialect<ondrix::ondsp::OndspDialect>();
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([&](UnrealizedConversionCastOp op) {
      return hasLegalConvertedTypes(op.getOperation(), typeConverter);
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             hasLegalConvertedTypes(op.getOperation(), typeConverter);
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return hasLegalConvertedTypes(op, typeConverter); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return hasLegalConvertedTypes(op, typeConverter); });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      bool hasLegalControlFlow =
          isNotBranchOpInterfaceOrReturnLikeOp(op) ||
          isLegalForBranchOpInterfaceTypeConversionPattern(op, typeConverter) ||
          isLegalForReturnOpTypeConversionPattern(op, typeConverter);
      return hasLegalControlFlow && hasLegalConvertedTypes(op, typeConverter);
    });

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspQ15ToScalarPass() {
  return std::make_unique<ConvertOndspQ15ToScalarPass>();
}
