#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPTOORTUMCORE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// The parameterless target type admits only this accumulator representation.
// Product semantics remain operation-specific legalization rules.
static bool isSupportedOrtumCoreAccumulator(ondrix::ondsp::AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.getWidth() == 40 && accumulator.getFrac() == 30 &&
         accumulator.getSignedness() == ondrix::ondsp::Signedness::Signed &&
         accumulator.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Saturate;
}

class OndspToOrtumCoreTypeConverter final : public TypeConverter {
public:
  explicit OndspToOrtumCoreTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](ondrix::ondsp::AccType type,
                            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      // Failing the specific conversion prevents the identity rule from
      // allowing an unsupported source accumulator to escape legalization.
      if (!isSupportedOrtumCoreAccumulator(type))
        return failure();
      results.push_back(ondrix::ortumcore::AccumType::get(context));
      return success();
    });
  }
};

static bool isScalarI32(Type type) { return type.isSignlessInteger(32); }

static LogicalResult verifyOrtumCoreAccumulator(Operation *op, ondrix::ondsp::AccType accumulator) {
  if (!isSupportedOrtumCoreAccumulator(accumulator))
    return op->emitOpError(
        "ortumcore lowering requires a signed 40-bit ondsp accumulator with frac=30 and "
        "saturating updates");
  return success();
}

static LogicalResult verifySupportedMacPolicy(Operation *op, ondrix::ondsp::FixedAttr numeric,
                                              ondrix::ondsp::ProductAttr product) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  if (ondrix::ondsp::isSignedQ15(numeric) && ondrix::ondsp::isFullProduct(product))
    return success();

  if (numeric.getSignedness() == ondrix::ondsp::Signedness::Signed && storage &&
      storage.isSignless() && storage.getWidth() == 32 && numeric.getFrac() == 31 &&
      product.getSelection() == ondrix::ondsp::ProductSelection::High)
    return op->emitOpError(
        "q31 high-product target equivalence is not specified; lower through a proven scalar "
        "sequence first");

  return op->emitOpError("ortumcore MAC lowering supports only signed q15 full-product semantics");
}

static bool containsOndspAccumulator(Type type) {
  // TypeConverter legality is shallow for aggregate types; reject source
  // accumulators recursively so tuples and other wrappers cannot leak through.
  return type.walk([](ondrix::ondsp::AccType) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool containsOndspAccumulator(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsOndspAccumulator(type); });
}

static ondrix::ondsp::AccType findUnsupportedAccumulator(Type type) {
  ondrix::ondsp::AccType unsupported;
  type.walk([&](ondrix::ondsp::AccType accumulator) {
    if (isSupportedOrtumCoreAccumulator(accumulator))
      return WalkResult::advance();
    unsupported = accumulator;
    return WalkResult::interrupt();
  });
  return unsupported;
}

static ondrix::ondsp::AccType findAccumulatorInAttribute(Attribute attribute) {
  ondrix::ondsp::AccType accumulator;
  attribute.walk([&](ondrix::ondsp::AccType type) {
    accumulator = type;
    return WalkResult::interrupt();
  });
  return accumulator;
}

static LogicalResult verifySupportedAccumulatorTypes(Operation *root) {
  WalkResult result = root->walk([&](Operation *op) {
    SmallVector<Type> types(op->getOperandTypes());
    llvm::append_range(types, op->getResultTypes());
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      llvm::append_range(types, function.getFunctionType().getInputs());
      llvm::append_range(types, function.getFunctionType().getResults());
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        llvm::append_range(types, block.getArgumentTypes());

    for (Type type : types) {
      ondrix::ondsp::AccType unsupported = findUnsupportedAccumulator(type);
      if (!unsupported)
        continue;
      op->emitOpError() << "unsupported accumulator type " << unsupported
                        << "; ortumcore lowering currently requires "
                           "!ondsp.acc<storage = i40, frac = 30, signed, "
                           "update_overflow = saturate>";
      return WalkResult::interrupt();
    }

    auto function = dyn_cast<func::FuncOp>(op);
    for (NamedAttribute namedAttribute : op->getAttrs()) {
      // Function signature types are converted structurally by the standard
      // function conversion patterns and were validated above.
      if (function && namedAttribute.getName() == function.getFunctionTypeAttrName())
        continue;
      ondrix::ondsp::AccType accumulator = findAccumulatorInAttribute(namedAttribute.getValue());
      if (!accumulator)
        continue;
      op->emitOpError() << "attribute '" << namedAttribute.getName().getValue()
                        << "' contains source accumulator type " << accumulator
                        << "; accumulator types in metadata attributes are unsupported";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  if (!typeConverter.isLegal(op) || containsOndspAccumulator(op->getOperandTypes()) ||
      containsOndspAccumulator(op->getResultTypes()))
    return false;
  if (llvm::any_of(op->getAttrs(), [](NamedAttribute namedAttribute) {
        return static_cast<bool>(findAccumulatorInAttribute(namedAttribute.getValue()));
      }))
    return false;
  for (Region &region : op->getRegions()) {
    if (!typeConverter.isLegal(&region))
      return false;
    for (Block &block : region) {
      if (containsOndspAccumulator(block.getArgumentTypes()))
        return false;
    }
  }
  return true;
}

class AccZeroOpLowering final : public OpConversionPattern<ondrix::ondsp::AccZeroOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccZeroOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getAcc().getType());
    if (!isa<ondrix::ortumcore::AccumType>(resultType))
      return op.emitError("ondsp.acc_zero lowering requires an accumulator result");

    rewriter.replaceOpWithNewOp<ondrix::ortumcore::AccInitOp>(op, resultType);
    return success();
  }
};

class AccImportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccImportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccImportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccImportOp op, OpAdaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError("exact accumulator import is unsupported by ortumcore lowering until "
                          "target import semantics are proven equivalent");
  }
};

class AccExportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccExportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccExportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExportOp op, OpAdaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError(
        "policy-bearing accumulator export is unsupported by ortumcore lowering until target "
        "export semantics are proven equivalent");
  }
};

class MacOpLowering final : public OpConversionPattern<ondrix::ondsp::MacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::MacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())) ||
        failed(verifySupportedMacPolicy(op, op.getNumeric(), op.getProduct())))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ortumcore::AccumType>(resultType))
      return op.emitOpError("MAC lowering requires converted target accumulator types");

    rewriter.replaceOpWithNewOp<ondrix::ortumcore::MacAddOp>(op, resultType, adaptor.getAcc(),
                                                             adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class MacSubOpLowering final : public OpConversionPattern<ondrix::ondsp::MacSubOp> {
public:
  using OpConversionPattern<ondrix::ondsp::MacSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())) ||
        failed(verifySupportedMacPolicy(op, op.getNumeric(), op.getProduct())))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ortumcore::AccumType>(resultType))
      return op.emitOpError("MAC-sub lowering requires converted target accumulator types");

    rewriter.replaceOpWithNewOp<ondrix::ortumcore::MacSubOp>(op, resultType, adaptor.getAcc(),
                                                             adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor,
                                ConversionPatternRewriter &) const override {
    if (!isa<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
      op.emitOpError("only fixed-point reduce_mac policies are supported by ortumcore lowering");
      return failure();
    }
    return op.emitOpError(
        "ordered fixed reduce_mac has no exact ortumcore lowering; scalarize to explicit "
        "accumulator updates or provide a proven target sequence");
  }
};

class CxButterflyOpLowering final : public OpConversionPattern<ondrix::ondsp::CxButterflyOp> {
public:
  using OpConversionPattern<ondrix::ondsp::CxButterflyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::CxButterflyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto fixed = dyn_cast_or_null<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (!fixed) {
      op.emitError("expected fixed numeric policy for ortumcore butterfly lowering");
      return failure();
    }

    auto intType = fixed.getStorage().dyn_cast<IntegerType>();
    if (fixed.getSignedness() != ondrix::ondsp::Signedness::Signed || !intType ||
        intType.getWidth() != 16 || fixed.getFrac() != 15) {
      op.emitError("only signed packed q15 butterfly lowering is supported");
      return failure();
    }

    auto layout = dyn_cast_or_null<ondrix::ondsp::CxLayoutAttr>(op.getLayout());
    if (!layout) {
      op.emitError("expected explicit complex layout for ortumcore butterfly lowering");
      return failure();
    }
    auto layoutValue = layout.getLayout();
    if (layoutValue != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo) {
      op.emitError("ortumcore lowering supports only packed_i16_imag_hi_real_lo layout");
      return failure();
    }

    for (Value operand : op.getOperands()) {
      if (!isScalarI32(operand.getType())) {
        op.emitError("ortumcore butterfly lowering requires signless scalar packed i32 operands");
        return failure();
      }
    }
    for (Value result : op.getResults()) {
      if (!isScalarI32(result.getType())) {
        op.emitError("ortumcore butterfly lowering requires signless scalar packed i32 results");
        return failure();
      }
    }

    auto scale = op.getScale();
    if (!scale)
      return op.emitOpError("packed q15 butterfly lowering requires an explicit scale policy");
    if (scale->getPreShiftLeft() != 0 || scale->getPostShiftRight() != 15 ||
        scale->getSaturateTo() != fixed.getStorage())
      return op.emitOpError("packed q15 butterfly lowering requires pre_shift_left=0, "
                            "post_shift_right=15, and saturate_to matching numeric storage");

    bool roundToNearest;
    switch (scale->getRounding()) {
    case ondrix::ondsp::RoundingMode::TowardNegative:
      roundToNearest = false;
      break;
    default:
      return op.emitOpError("ortumcore butterfly lowering supports toward_negative rounding");
    }

    bool saturation;
    switch (scale->getOverflow()) {
    case ondrix::ondsp::OverflowMode::Wrap:
      saturation = false;
      break;
    case ondrix::ondsp::OverflowMode::Saturate:
      saturation = true;
      break;
    default:
      return op.emitOpError("ortumcore butterfly lowering supports wrap or saturate overflow");
    }

    auto product = op.getProduct();
    if (!product || product->getSelection() != ondrix::ondsp::ProductSelection::Full)
      return op.emitOpError(
          "packed q15 butterfly lowering requires product = #ondsp.product<full>");

    if (op.getTrivialTwiddle())
      return op.emitOpError("trivial-twiddle target selection is disabled until the twiddle "
                            "value, stage role, layout, permutation, and scale are proven");

    Type out0Type = getTypeConverter()->convertType(op.getOut0().getType());
    Type out1Type = getTypeConverter()->convertType(op.getOut1().getType());
    Type stateType = ondrix::ortumcore::VecStateType::get(rewriter.getContext());
    auto init = rewriter.create<ondrix::ortumcore::VecStateInitOp>(op.getLoc(), stateType);
    auto mode = rewriter.create<ondrix::ortumcore::VecSetModeOp>(
        op.getLoc(), stateType, init.getState(), rewriter.getBoolAttr(saturation),
        rewriter.getBoolAttr(roundToNearest), rewriter.getBoolAttr(true),
        rewriter.getI64IntegerAttr(scale->getPostShiftRight()),
        rewriter.getI64IntegerAttr(scale->getPreShiftLeft()));

    auto mul = rewriter.create<ondrix::ortumcore::CxMulOp>(
        op.getLoc(), stateType, adaptor.getB().getType(), mode.getResult(), adaptor.getB(),
        adaptor.getTwiddle());
    auto add = rewriter.create<ondrix::ortumcore::CxDualAddOp>(
        op.getLoc(), stateType, out0Type, mul.getNextState(), adaptor.getA(), mul.getResult());
    auto sub = rewriter.create<ondrix::ortumcore::CxDualSubOp>(
        op.getLoc(), stateType, out1Type, add.getNextState(), adaptor.getA(), mul.getResult());
    rewriter.replaceOp(op, ValueRange{add.getResult(), sub.getResult()});
    return success();
  }
};

class ConvertOndspToOrtumCorePass final
    : public ondrix::impl::ConvertOndspToOrtumCoreBase<ConvertOndspToOrtumCorePass> {
public:
  using ondrix::impl::ConvertOndspToOrtumCoreBase<
      ConvertOndspToOrtumCorePass>::ConvertOndspToOrtumCoreBase;

  void runOnOperation() override {
    if (failed(verifySupportedAccumulatorTypes(getOperation()))) {
      signalPassFailure();
      return;
    }

    OndspToOrtumCoreTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AccZeroOpLowering, AccImportOpLowering, AccExportOpLowering, MacOpLowering,
                 MacSubOpLowering, ReduceMacOpLowering, CxButterflyOpLowering>(typeConverter,
                                                                               &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, ondrix::ortumcore::OrtumCoreDialect>();
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([&](UnrealizedConversionCastOp op) {
      return hasLegalConvertedTypes(op.getOperation(), typeConverter);
    });
    target.addIllegalDialect<ondrix::ondsp::OndspDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             !containsOndspAccumulator(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) && hasLegalConvertedTypes(op, typeConverter);
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

std::unique_ptr<Pass> ondrix::createConvertOndspToOrtumCorePass() {
  return std::make_unique<ConvertOndspToOrtumCorePass>();
}
