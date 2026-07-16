#include "ondrix/Conversion/OrtumCoreToOndsp/OrtumCoreToOndsp.h"
#include "ondrix/Conversion/Utils/StructuralTypeConversions.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTORTUMCORETOONDSPEMULATION
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static ondrix::ondsp::AccType getEmulatedAccumulatorType(MLIRContext *context) {
  return ondrix::ondsp::AccType::get(context, IntegerType::get(context, 40), 30,
                                     ondrix::ondsp::Signedness::Signed,
                                     ondrix::ondsp::OverflowMode::Saturate);
}

class OrtumCoreToOndspTypeConverter final : public TypeConverter {
public:
  explicit OrtumCoreToOndspTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](ondrix::ortumcore::AccumType,
                            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      results.push_back(getEmulatedAccumulatorType(context));
      return success();
    });
  }
};

static bool containsOrtumCoreAccumulator(Type type) {
  return type.walk([](ondrix::ortumcore::AccumType) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool containsOrtumCoreAccumulator(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsOrtumCoreAccumulator(type); });
}

static bool containsOrtumCoreAccumulator(Attribute attribute) {
  return attribute.walk([](ondrix::ortumcore::AccumType) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool isNestedAccumulatorContainer(Type type) {
  return !isa<ondrix::ortumcore::AccumType>(type) && containsOrtumCoreAccumulator(type);
}

static LogicalResult verifyAccumulatorUsage(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    auto containsNested = [](TypeRange types) {
      return llvm::any_of(types, isNestedAccumulatorContainer);
    };
    if (containsNested(op->getOperandTypes()) || containsNested(op->getResultTypes())) {
      op->emitOpError("nested OrtumCore accumulator containers are unsupported");
      return WalkResult::interrupt();
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      FunctionType type = function.getFunctionType();
      if (containsNested(type.getInputs()) || containsNested(type.getResults())) {
        op->emitOpError("nested OrtumCore accumulator containers are unsupported");
        return WalkResult::interrupt();
      }
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (containsNested(block.getArgumentTypes())) {
          op->emitOpError("nested OrtumCore accumulator containers are unsupported");
          return WalkResult::interrupt();
        }

    auto function = dyn_cast<func::FuncOp>(op);
    for (NamedAttribute namedAttribute : op->getAttrs()) {
      if (function && namedAttribute.getName() == function.getFunctionTypeAttrName())
        continue;
      if (!containsOrtumCoreAccumulator(namedAttribute.getValue()))
        continue;
      op->emitOpError() << "attribute '" << namedAttribute.getName().getValue()
                        << "' contains a target accumulator type; accumulator types in metadata "
                           "attributes are unsupported";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  if (!typeConverter.isLegal(op) || containsOrtumCoreAccumulator(op->getOperandTypes()) ||
      containsOrtumCoreAccumulator(op->getResultTypes()))
    return false;
  if (llvm::any_of(op->getAttrs(), [](NamedAttribute namedAttribute) {
        return containsOrtumCoreAccumulator(namedAttribute.getValue());
      }))
    return false;
  for (Region &region : op->getRegions()) {
    if (!typeConverter.isLegal(&region))
      return false;
    for (Block &block : region)
      if (containsOrtumCoreAccumulator(block.getArgumentTypes()))
        return false;
  }
  return true;
}

class AccInitOpLowering final : public OpConversionPattern<ondrix::ortumcore::AccInitOp> {
public:
  using OpConversionPattern<ondrix::ortumcore::AccInitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ortumcore::AccInitOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ondsp::AccType>(resultType))
      return op.emitOpError("emulation requires a converted Ondsp accumulator result");
    rewriter.replaceOpWithNewOp<ondrix::ondsp::AccZeroOp>(op, resultType);
    return success();
  }
};

template <typename SourceOp, typename DestinationOp>
class MacLikeOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ondsp::AccType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ondsp::AccType>(resultType))
      return op.emitOpError("emulation requires converted Ondsp accumulator types");

    auto numeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), ondrix::ondsp::Signedness::Signed, rewriter.getI16Type(), 15);
    auto product = ondrix::ondsp::ProductAttr::get(rewriter.getContext(),
                                                   ondrix::ondsp::ProductSelection::Full);
    rewriter.replaceOpWithNewOp<DestinationOp>(op, resultType, adaptor.getAcc(), adaptor.getLhs(),
                                               adaptor.getRhs(), numeric, product);
    return success();
  }
};

using MacAddOpLowering = MacLikeOpLowering<ondrix::ortumcore::MacAddOp, ondrix::ondsp::MacOp>;
using MacSubOpLowering = MacLikeOpLowering<ondrix::ortumcore::MacSubOp, ondrix::ondsp::MacSubOp>;

class ConvertOrtumCoreToOndspEmulationPass final
    : public ondrix::impl::ConvertOrtumCoreToOndspEmulationBase<
          ConvertOrtumCoreToOndspEmulationPass> {
public:
  using ondrix::impl::ConvertOrtumCoreToOndspEmulationBase<
      ConvertOrtumCoreToOndspEmulationPass>::ConvertOrtumCoreToOndspEmulationBase;

  void runOnOperation() override {
    if (failed(verifyAccumulatorUsage(getOperation()))) {
      signalPassFailure();
      return;
    }

    OrtumCoreToOndspTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AccInitOpLowering, MacAddOpLowering, MacSubOpLowering>(typeConverter,
                                                                        &getContext());
    ondrix::conversion::populateCommonStructuralTypeConversionPatterns(typeConverter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, ondrix::ondsp::OndspDialect>();
    target.addIllegalDialect<ondrix::ortumcore::OrtumCoreDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](UnrealizedConversionCastOp op) {
      return !containsOrtumCoreAccumulator(op.getOperandTypes()) &&
             !containsOrtumCoreAccumulator(op.getResultTypes());
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

std::unique_ptr<Pass> ondrix::createConvertOrtumCoreToOndspEmulationPass() {
  return std::make_unique<ConvertOrtumCoreToOndspEmulationPass>();
}
