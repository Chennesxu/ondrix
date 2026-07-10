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

using namespace mlir;

namespace {

class OndspToOrtumCoreTypeConverter final : public TypeConverter {
public:
  explicit OndspToOrtumCoreTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](ondrix::ondsp::AccType) -> Type {
      return ondrix::ortumcore::AccumType::get(context);
    });
  }
};

enum class MacTarget {
  DualMac,
  QMacAdd,
};

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

static LogicalResult verifyStorageOperand(Operation *op, ondrix::ondsp::FixedAttr fixed,
                                          Type type) {
  if (!hasStorageType(type, fixed.getStorage()))
    return op->emitError("operand type does not match fixed numeric storage type");
  return success();
}

static FailureOr<MacTarget> chooseMacTarget(Attribute numeric) {
  auto fixed = dyn_cast_or_null<ondrix::ondsp::FixedAttr>(numeric);
  if (!fixed || fixed.getSignedness() != ondrix::ondsp::Signedness::Signed)
    return failure();

  auto intType = fixed.getStorage().dyn_cast<IntegerType>();
  if (!intType)
    return failure();

  if (intType.getWidth() == 16 && fixed.getFrac() == 15)
    return MacTarget::DualMac;
  if (intType.getWidth() == 32 && fixed.getFrac() == 31)
    return MacTarget::QMacAdd;
  return failure();
}

class AccInitOpLowering final : public OpConversionPattern<ondrix::ondsp::AccInitOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccInitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccInitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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
    FailureOr<MacTarget> target = chooseMacTarget(op.getNumeric());
    if (failed(target)) {
      op.emitError("only signed q15 and signed q31 MAC are supported by ortumcore lowering");
      return failure();
    }

    auto fixed = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
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

    switch (*target) {
    case MacTarget::DualMac:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::DualMacOp>(op, resultType, adaptor.getAcc(),
                                                                adaptor.getLhs(), adaptor.getRhs());
      return success();
    case MacTarget::QMacAdd:
      rewriter.replaceOpWithNewOp<ondrix::ortumcore::QMacAddOp>(op, resultType, adaptor.getAcc(),
                                                                adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    llvm_unreachable("unhandled MAC target");
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<MacTarget> target = chooseMacTarget(op.getNumeric());
    if (failed(target)) {
      op.emitError("only signed q15 and signed q31 reduce_mac are supported by ortumcore lowering");
      return failure();
    }

    auto fixed = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (failed(verifyStorageOperand(op, fixed, op.getLhs().getType())) ||
        failed(verifyStorageOperand(op, fixed, op.getRhs().getType())))
      return failure();

    Type accType = ondrix::ortumcore::AccumType::get(rewriter.getContext());
    auto init = rewriter.create<ondrix::ortumcore::AccInitOp>(op.getLoc(), accType);
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    Value accumulated;
    switch (*target) {
    case MacTarget::DualMac:
      accumulated =
          rewriter
              .create<ondrix::ortumcore::DualMacOp>(op.getLoc(), accType, init.getResult(),
                                                    adaptor.getLhs(), adaptor.getRhs())
              .getResult();
      break;
    case MacTarget::QMacAdd:
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
    if (layoutValue != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo &&
        layoutValue != ondrix::ondsp::ComplexLayout::PackedI16RealHiImagLo) {
      op.emitError("only packed i16 complex layouts are supported");
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

    Type out0Type = getTypeConverter()->convertType(op.getOut0().getType());
    Type out1Type = getTypeConverter()->convertType(op.getOut1().getType());
    Type stateType = ondrix::ortumcore::VecStateType::get(rewriter.getContext());
    auto init = rewriter.create<ondrix::ortumcore::VecStateInitOp>(op.getLoc(), stateType);
    auto mode =
        rewriter.create<ondrix::ortumcore::VecSetModeOp>(op.getLoc(), stateType, init.getState());
    mode->setAttr("sat", rewriter.getBoolAttr(true));
    mode->setAttr("rnd", rewriter.getBoolAttr(false));
    mode->setAttr("pack", rewriter.getBoolAttr(true));
    mode->setAttr("shiftr", rewriter.getI64IntegerAttr(15));
    mode->setAttr("shiftl", rewriter.getI64IntegerAttr(0));

    auto trivialTwiddle = op->getAttrOfType<BoolAttr>("trivial_twiddle");
    if (trivialTwiddle && trivialTwiddle.getValue()) {
      auto fft = rewriter.create<ondrix::ortumcore::FftTrivialStageOp>(
          op.getLoc(), stateType, TypeRange{out0Type, out1Type}, mode.getResult(),
          ValueRange{adaptor.getA(), adaptor.getB()},
          ondrix::ortumcore::FftStageKindAttr::get(rewriter.getContext(),
                                                   ondrix::ortumcore::FftStageKind::Radix2));
      rewriter.replaceOp(op, fft.getOutputs());
      return success();
    }

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

class ConvertOndspToOrtumCorePass
    : public PassWrapper<ConvertOndspToOrtumCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOndspToOrtumCorePass)

  StringRef getArgument() const final { return "convert-ondsp-to-ortumcore"; }
  StringRef getDescription() const final {
    return "Lower ondsp semantic ops to ortumcore target semantic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ondrix::ondsp::OndspDialect, ondrix::ortumcore::OrtumCoreDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    OndspToOrtumCoreTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AccInitOpLowering, AccExtractOpLowering, MacOpLowering, ReduceMacOpLowering,
                 CxButterflyOpLowering>(typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, ondrix::ortumcore::OrtumCoreDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<ondrix::ondsp::OndspDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isSignatureLegal(op.getCalleeType()); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspToOrtumCorePass() {
  return std::make_unique<ConvertOndspToOrtumCorePass>();
}

void ondrix::registerConvertOndspToOrtumCorePass() {
  PassRegistration<ConvertOndspToOrtumCorePass>();
}
