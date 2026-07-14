#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

class OndspToOrtumCoreTypeConverter final : public TypeConverter {
public:
  explicit OndspToOrtumCoreTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](ondrix::ondsp::AccType type,
                            SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      // Failing the specific conversion prevents the identity rule from
      // allowing an unsupported source accumulator to escape legalization.
      auto storage = type.getStorage().dyn_cast<IntegerType>();
      if (!storage || storage.getWidth() != 40 ||
          type.getSignedness() != ondrix::ondsp::Signedness::Signed)
        return failure();
      results.push_back(ondrix::ortumcore::AccumType::get(context));
      return success();
    });
  }
};

enum class MacLoweringKind { Mac, QMac };

static bool hasStorageType(Type type, Type storage) {
  if (type == storage)
    return true;

  if (auto shaped = type.dyn_cast<ShapedType>())
    return shaped.getElementType() == storage;

  return false;
}

static bool isI32(Type type) {
  auto intType = type.dyn_cast<IntegerType>();
  return intType && intType.getWidth() == 32;
}

static LogicalResult verifyOrtumCoreAccumulator(Operation *op, ondrix::ondsp::AccType accumulator) {
  auto storage = accumulator.getStorage().dyn_cast<IntegerType>();
  if (!storage || storage.getWidth() != 40 ||
      accumulator.getSignedness() != ondrix::ondsp::Signedness::Signed)
    return op->emitOpError("ortumcore lowering requires a signed 40-bit ondsp accumulator");
  return success();
}

static bool containsOndspAccumulator(Type type) {
  // TypeConverter legality is shallow for aggregate types; reject source
  // accumulators recursively so tuples and other wrappers cannot leak through.
  return type.walk([](ondrix::ondsp::AccType) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool containsOndspAccumulator(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsOndspAccumulator(type); });
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  if (!typeConverter.isLegal(op) || containsOndspAccumulator(op->getOperandTypes()) ||
      containsOndspAccumulator(op->getResultTypes()))
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

static LogicalResult verifyStorageOperand(Operation *op, ondrix::ondsp::FixedAttr fixed,
                                          Type type) {
  if (!hasStorageType(type, fixed.getStorage()))
    return op->emitError("operand type does not match fixed numeric storage type");
  return success();
}

static LogicalResult selectMacLoweringKind(Operation *op, ondrix::ondsp::FixedAttr fixed,
                                           std::optional<ondrix::ondsp::ProductAttr> product,
                                           MacLoweringKind &kind) {
  if (!product)
    return op->emitOpError("lowering requires an explicit fixed-point product policy");

  auto intType = fixed.getStorage().dyn_cast<IntegerType>();
  if (fixed.getSignedness() != ondrix::ondsp::Signedness::Signed || !intType)
    return op->emitOpError(
        "only signed q15/product=full and signed q31/product=high MAC policies are supported");

  if (intType.getWidth() == 16 && fixed.getFrac() == 15) {
    if (product->getSelection() != ondrix::ondsp::ProductSelection::Full)
      return op->emitOpError("signed q15 MAC lowering requires product = #ondsp.product<full>");
    kind = MacLoweringKind::Mac;
    return success();
  }

  if (intType.getWidth() == 32 && fixed.getFrac() == 31) {
    if (product->getSelection() != ondrix::ondsp::ProductSelection::High)
      return op->emitOpError("signed q31 MAC lowering requires product = #ondsp.product<high>");
    kind = MacLoweringKind::QMac;
    return success();
  }

  return op->emitOpError(
      "only signed q15/product=full and signed q31/product=high MAC policies are supported");
}

class AccInitOpLowering final : public OpConversionPattern<ondrix::ondsp::AccInitOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccInitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccInitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())))
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getAcc().getType());
    if (!isa<ondrix::ortumcore::AccumType>(resultType))
      return op.emitError("ondsp.acc_init lowering requires an accumulator result");

    rewriter.replaceOpWithNewOp<ondrix::ortumcore::AccImportOp>(op, resultType, adaptor.getInput());
    return success();
  }
};

class AccExtractOpLowering final : public OpConversionPattern<ondrix::ondsp::AccExtractOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())))
      return failure();
    if (op.getScale())
      return op.emitError("scaled ondsp.acc_extract is not supported by ortumcore lowering");
    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()))
      return op.emitError("ondsp.acc_extract lowering requires an accumulator operand");

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<ondrix::ortumcore::AccExtractOp>(op, resultType, adaptor.getAcc());
    return success();
  }
};

class MacOpLowering final : public OpConversionPattern<ondrix::ondsp::MacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::MacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    MacLoweringKind kind;
    if (failed(selectMacLoweringKind(op, op.getNumeric(), op.getProduct(), kind)))
      return failure();
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())))
      return failure();

    auto fixed = op.getNumeric();
    if (failed(verifyStorageOperand(op, fixed, op.getLhs().getType())) ||
        failed(verifyStorageOperand(op, fixed, op.getRhs().getType())))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ortumcore::AccumType>(resultType)) {
      op.emitError(
          "standalone ondsp.mac lowering requires !ortumcore.acc accumulator operands/results");
      return failure();
    }

    switch (kind) {
    case MacLoweringKind::Mac:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::MacAddOp>(op, resultType, adaptor.getAcc(),
                                                               adaptor.getLhs(), adaptor.getRhs());
      return success();
    case MacLoweringKind::QMac:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::QMacAddOp>(op, resultType, adaptor.getAcc(),
                                                                adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    llvm_unreachable("unhandled MAC lowering kind");
  }
};

class MacSubOpLowering final : public OpConversionPattern<ondrix::ondsp::MacSubOp> {
public:
  using OpConversionPattern<ondrix::ondsp::MacSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    MacLoweringKind kind;
    if (failed(selectMacLoweringKind(op, op.getNumeric(), op.getProduct(), kind)))
      return failure();
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())))
      return failure();

    auto fixed = op.getNumeric();
    if (failed(verifyStorageOperand(op, fixed, op.getLhs().getType())) ||
        failed(verifyStorageOperand(op, fixed, op.getRhs().getType())))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ortumcore::AccumType>(resultType)) {
      op.emitError(
          "standalone ondsp.mac_sub lowering requires !ortumcore.acc accumulator operands/results");
      return failure();
    }

    switch (kind) {
    case MacLoweringKind::Mac:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::MacSubOp>(op, resultType, adaptor.getAcc(),
                                                               adaptor.getLhs(), adaptor.getRhs());
      return success();
    case MacLoweringKind::QMac:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::QMacSubOp>(op, resultType, adaptor.getAcc(),
                                                                adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    llvm_unreachable("unhandled MAC lowering kind");
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getLhs().getType()) || isa<ShapedType>(op.getRhs().getType())) {
      op.emitOpError("shaped operands are not supported by ortumcore lowering; lower the "
                     "reduction to an explicit loop of scalar MAC operations first");
      return failure();
    }

    auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (!fixed) {
      op.emitOpError("only fixed-point reduce_mac policies are supported by ortumcore lowering");
      return failure();
    }

    MacLoweringKind kind;
    if (failed(selectMacLoweringKind(op, fixed, op.getProduct(), kind)))
      return failure();

    if (failed(verifyStorageOperand(op, fixed, op.getLhs().getType())) ||
        failed(verifyStorageOperand(op, fixed, op.getRhs().getType())))
      return failure();

    Type accType = ondrix::ortumcore::AccumType::get(rewriter.getContext());
    auto init = rewriter.create<ondrix::ortumcore::AccInitOp>(op.getLoc(), accType);
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    Value accumulated;
    switch (kind) {
    case MacLoweringKind::Mac:
      accumulated = rewriter
                        .create<ondrix::ortumcore::MacAddOp>(op.getLoc(), accType, init.getResult(),
                                                             adaptor.getLhs(), adaptor.getRhs())
                        .getResult();
      break;
    case MacLoweringKind::QMac:
      accumulated =
          rewriter
              .create<ondrix::ortumcore::QMacAddOp>(op.getLoc(), accType, init.getResult(),
                                                    adaptor.getLhs(), adaptor.getRhs())
              .getResult();
      break;
    }

    auto extract =
        rewriter.create<ondrix::ortumcore::AccExtractOp>(op.getLoc(), resultType, accumulated);
    rewriter.replaceOp(op, extract.getResult());
    return success();
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
      if (!isI32(operand.getType())) {
        op.emitError("packed q15 butterfly operands must use i32 storage");
        return failure();
      }
    }
    for (Value result : op.getResults()) {
      if (!isI32(result.getType())) {
        op.emitError("packed q15 butterfly results must use i32 storage");
        return failure();
      }
    }

    auto scale = op.getScale();
    if (!scale)
      return op.emitOpError("packed q15 butterfly lowering requires an explicit scale policy");
    if (scale->getPreShiftLeft() != 0 || scale->getPostShiftRight() != 15 ||
        !scale->getSaturateTo().isInteger(16))
      return op.emitOpError("packed q15 butterfly lowering requires pre_shift_left=0, "
                            "post_shift_right=15, and saturate_to=i16");

    bool rounding;
    switch (scale->getRounding()) {
    case ondrix::ondsp::RoundingMode::Trunc:
      rounding = false;
      break;
    case ondrix::ondsp::RoundingMode::Nearest:
      rounding = true;
      break;
    default:
      return op.emitOpError("ortumcore butterfly lowering supports trunc or nearest rounding");
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
    auto mode =
        rewriter.create<ondrix::ortumcore::VecSetModeOp>(op.getLoc(), stateType, init.getState());
    mode->setAttr("sat", rewriter.getBoolAttr(saturation));
    mode->setAttr("rnd", rewriter.getBoolAttr(rounding));
    mode->setAttr("pack", rewriter.getBoolAttr(true));
    mode->setAttr("shiftr", rewriter.getI64IntegerAttr(scale->getPostShiftRight()));
    mode->setAttr("shiftl", rewriter.getI64IntegerAttr(scale->getPreShiftLeft()));

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
    OndspToOrtumCoreTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AccInitOpLowering, AccExtractOpLowering, MacOpLowering, MacSubOpLowering,
                 ReduceMacOpLowering, CxButterflyOpLowering>(typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, ondrix::ortumcore::OrtumCoreDialect>();
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) { return typeConverter.isLegal(op); });
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
