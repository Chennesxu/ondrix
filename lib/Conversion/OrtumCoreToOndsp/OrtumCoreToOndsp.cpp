#include "ondrix/Conversion/OrtumCoreToOndsp/OrtumCoreToOndsp.h"
#include "ondrix/Conversion/Utils/ConversionLegality.h"
#include "ondrix/Conversion/Utils/ValueTypeConversions.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"
#include "ondrix/Target/OrtumCore/OrtumCoreCapabilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/AttrTypeSubElements.h"
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
  ondrix::ortumcore::AccumulatorDomain domain =
      ondrix::ortumcore::getSignedI40Frac30SaturatingAccumulatorDomain();
  return ondrix::ondsp::AccType::get(context, IntegerType::get(context, domain.storageWidth),
                                     domain.frac, domain.signedness, domain.updateOverflow);
}

static bool isOrtumCoreType(Type type) {
  return type.getDialect().getNamespace() ==
         ondrix::ortumcore::OrtumCoreDialect::getDialectNamespace();
}

static bool isOrtumCoreAttribute(Attribute attribute) {
  return attribute.getDialect().getNamespace() ==
         ondrix::ortumcore::OrtumCoreDialect::getDialectNamespace();
}

class OrtumCoreToOndspTypeConverter final : public TypeConverter {
public:
  explicit OrtumCoreToOndspTypeConverter(MLIRContext *context) {
    addConversion([](Type type, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (isOrtumCoreType(type))
        return failure();
      results.push_back(type);
      return success();
    });
    addConversion([context](ondrix::ortumcore::AccumType,
                            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      results.push_back(getEmulatedAccumulatorType(context));
      return success();
    });
  }
};

static bool containsOrtumCoreType(Type type) {
  return ondrix::conversion::containsMatchingType(type, isOrtumCoreType);
}

static bool containsOrtumCoreType(TypeRange types) {
  return ondrix::conversion::containsMatchingType(types, isOrtumCoreType);
}

static bool containsOrtumCoreArtifact(Attribute attribute) {
  return ondrix::conversion::containsMatchingType(attribute, isOrtumCoreType) ||
         ondrix::conversion::containsMatchingAttribute(attribute, isOrtumCoreAttribute);
}

static bool isUnsupportedOrNestedOrtumCoreType(Type type) {
  return !isa<ondrix::ortumcore::AccumType>(type) && containsOrtumCoreType(type);
}

static LogicalResult verifySourceArtifactUsage(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    auto containsUnsupported = [](TypeRange types) {
      return llvm::any_of(types, isUnsupportedOrNestedOrtumCoreType);
    };
    if (containsUnsupported(op->getOperandTypes()) || containsUnsupported(op->getResultTypes())) {
      op->emitOpError("unsupported or nested OrtumCore type");
      return WalkResult::interrupt();
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      FunctionType type = function.getFunctionType();
      if (containsUnsupported(type.getInputs()) || containsUnsupported(type.getResults())) {
        op->emitOpError("unsupported or nested OrtumCore type");
        return WalkResult::interrupt();
      }
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (containsUnsupported(block.getArgumentTypes())) {
          op->emitOpError("unsupported or nested OrtumCore type");
          return WalkResult::interrupt();
        }

    auto function = dyn_cast<func::FuncOp>(op);
    for (NamedAttribute namedAttribute : op->getAttrs()) {
      if (function && namedAttribute.getName() == function.getFunctionTypeAttrName())
        continue;
      if (!containsOrtumCoreArtifact(namedAttribute.getValue()))
        continue;
      op->emitOpError() << "attribute '" << namedAttribute.getName().getValue()
                        << "' contains an OrtumCore type or attribute; target artifacts in "
                           "metadata attributes are unsupported";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  return ondrix::conversion::hasLegalConvertedTypesAndAttributes(op, typeConverter, isOrtumCoreType,
                                                                 isOrtumCoreAttribute);
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

    ondrix::ortumcore::ProductDomain domain = ondrix::ortumcore::getSignedQ15FullProductDomain();
    auto numeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), domain.signedness,
        IntegerType::get(rewriter.getContext(), domain.operandWidth), domain.operandFrac);
    auto product = ondrix::ondsp::ProductAttr::get(rewriter.getContext(), domain.product.selection);
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
    if (failed(verifySourceArtifactUsage(getOperation()))) {
      signalPassFailure();
      return;
    }
    if (failed(ondrix::conversion::verifySCFWhileTypeConversionSafety(
            getOperation(), [](Type type) { return isa<ondrix::ortumcore::AccumType>(type); }))) {
      signalPassFailure();
      return;
    }

    OrtumCoreToOndspTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AccInitOpLowering, MacAddOpLowering, MacSubOpLowering>(typeConverter,
                                                                        &getContext());
    ondrix::conversion::populateValueTypeConversionPatterns(typeConverter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<ondrix::ondsp::OndspDialect>(
        [&](Operation *op) { return hasLegalConvertedTypes(op, typeConverter); });
    target.addIllegalDialect<ondrix::ortumcore::OrtumCoreDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](UnrealizedConversionCastOp op) {
      return !containsOrtumCoreType(op.getOperandTypes()) &&
             !containsOrtumCoreType(op.getResultTypes());
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
