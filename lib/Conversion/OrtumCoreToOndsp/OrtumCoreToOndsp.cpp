#include "ondrix/Conversion/OrtumCoreToOndsp/OrtumCoreToOndsp.h"
#include "ondrix/Conversion/Utils/ConversionLegality.h"
#include "ondrix/Conversion/Utils/ValueTypeConversions.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
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

/// Ondsp operations stay legal in this pass, so a multi-lane accumulator
/// already present in the module would pass through untouched next to
/// emulated single-lane target accumulators. The emulation profile has no
/// multi-lane form, so reject the whole module instead of mixing the two lane
/// meanings silently.
static LogicalResult verifySingleLaneOndspAccumulators(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    SmallVector<Type> types;
    ondrix::ondsp::appendAccumulatorCandidateTypes(op, types);

    for (Type type : types) {
      ondrix::ondsp::AccType multiLane =
          ondrix::ondsp::findRejectedAccumulator(type, ondrix::ondsp::isSingleLaneAccumulator);
      if (!multiLane)
        continue;
      op->emitOpError() << "multi-lane accumulator type " << multiLane
                        << " has no OrtumCore emulation profile";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
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

class DmacOpLowering final : public OpConversionPattern<ondrix::ortumcore::DmacOp> {
public:
  using OpConversionPattern<ondrix::ortumcore::DmacOp>::OpConversionPattern;

  // The dual step denotes exactly two independent mac_add updates, so the
  // emulation is literally two ondsp.mac operations, one per lane.
  LogicalResult matchAndRewrite(ondrix::ortumcore::DmacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type lane0Type = this->getTypeConverter()->convertType(op.getOut0().getType());
    Type lane1Type = this->getTypeConverter()->convertType(op.getOut1().getType());
    if (!isa_and_nonnull<ondrix::ondsp::AccType>(lane0Type) ||
        !isa_and_nonnull<ondrix::ondsp::AccType>(lane1Type) ||
        !isa<ondrix::ondsp::AccType>(adaptor.getLane0().getType()) ||
        !isa<ondrix::ondsp::AccType>(adaptor.getLane1().getType()))
      return op.emitOpError("emulation requires converted Ondsp accumulator types");

    ondrix::ortumcore::ProductDomain domain = ondrix::ortumcore::getSignedQ15FullProductDomain();
    auto numeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), domain.signedness,
        IntegerType::get(rewriter.getContext(), domain.operandWidth), domain.operandFrac);
    auto product = ondrix::ondsp::ProductAttr::get(rewriter.getContext(), domain.product.selection);
    Value lane0 = rewriter.create<ondrix::ondsp::MacOp>(op.getLoc(), lane0Type, adaptor.getLane0(),
                                                        adaptor.getLhs0(), adaptor.getRhs0(),
                                                        numeric, product);
    Value lane1 = rewriter.create<ondrix::ondsp::MacOp>(op.getLoc(), lane1Type, adaptor.getLane1(),
                                                        adaptor.getLhs1(), adaptor.getRhs1(),
                                                        numeric, product);
    rewriter.replaceOp(op, {lane0, lane1});
    return success();
  }
};

class AccOutOpLowering final : public OpConversionPattern<ondrix::ortumcore::AccOutOp> {
public:
  using OpConversionPattern<ondrix::ortumcore::AccOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ortumcore::AccOutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = dyn_cast<ondrix::ondsp::AccType>(adaptor.getAcc().getType());
    if (!accumulator)
      return op.emitOpError("emulation requires a converted Ondsp accumulator operand");
    // acc_export with an i32 destination realizes the readout equation
    // exactly: floor division by 2^shift, then saturation at the i32 storage.
    ondrix::ortumcore::ExportDomain domain =
        ondrix::ortumcore::getShiftedSaturatingI32ExportDomain();
    auto destination =
        ondrix::ondsp::FixedAttr::get(rewriter.getContext(), accumulator.getSignedness(),
                                      IntegerType::get(rewriter.getContext(), domain.storageWidth),
                                      accumulator.getFrac() - unsigned(op.getShift()));
    rewriter.replaceOpWithNewOp<ondrix::ondsp::AccExportOp>(
        op, rewriter.getI32Type(), adaptor.getAcc(), destination, domain.rounding, domain.overflow);
    return success();
  }
};

// rev32 as the standard five-step masked butterfly swap; kept in plain
// arith so every downstream generic path can execute it.
static Value emitReverse32(OpBuilder &builder, Location loc, Value value) {
  auto swapStep = [&](Value input, int32_t mask, int32_t amount) -> Value {
    Value maskValue = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(mask));
    Value amountValue = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(amount));
    Value right = builder.create<arith::ShRUIOp>(loc, input, amountValue);
    right = builder.create<arith::AndIOp>(loc, right, maskValue);
    Value left = builder.create<arith::AndIOp>(loc, input, maskValue);
    left = builder.create<arith::ShLIOp>(loc, left, amountValue);
    return builder.create<arith::OrIOp>(loc, right, left);
  };
  value = swapStep(value, 0x55555555, 1);
  value = swapStep(value, 0x33333333, 2);
  value = swapStep(value, 0x0F0F0F0F, 4);
  value = swapStep(value, 0x00FF00FF, 8);
  Value sixteen = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(16));
  Value high = builder.create<arith::ShRUIOp>(loc, value, sixteen);
  Value low = builder.create<arith::ShLIOp>(loc, value, sixteen);
  return builder.create<arith::OrIOp>(loc, high, low);
}

template <typename SourceOp, bool Subtract>
class BitrevLikeOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                typename OpConversionPattern<SourceOp>::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = emitReverse32(rewriter, loc, adaptor.getLhs());
    Value rhs = emitReverse32(rewriter, loc, adaptor.getRhs());
    Value raw = Subtract ? rewriter.create<arith::SubIOp>(loc, lhs, rhs).getResult()
                         : rewriter.create<arith::AddIOp>(loc, lhs, rhs).getResult();
    rewriter.replaceOp(op, emitReverse32(rewriter, loc, raw));
    return success();
  }
};

using BitrevAddOpLowering = BitrevLikeOpLowering<ondrix::ortumcore::BitrevAddOp, false>;
using BitrevSubOpLowering = BitrevLikeOpLowering<ondrix::ortumcore::BitrevSubOp, true>;

class ConvertOrtumCoreToOndspEmulationPass final
    : public ondrix::impl::ConvertOrtumCoreToOndspEmulationBase<
          ConvertOrtumCoreToOndspEmulationPass> {
public:
  using ondrix::impl::ConvertOrtumCoreToOndspEmulationBase<
      ConvertOrtumCoreToOndspEmulationPass>::ConvertOrtumCoreToOndspEmulationBase;

  void runOnOperation() override {
    if (failed(verifySourceArtifactUsage(getOperation())) ||
        failed(verifySingleLaneOndspAccumulators(getOperation()))) {
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
    patterns.add<AccInitOpLowering, MacAddOpLowering, MacSubOpLowering, DmacOpLowering,
                 AccOutOpLowering, BitrevAddOpLowering, BitrevSubOpLowering>(typeConverter,
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
