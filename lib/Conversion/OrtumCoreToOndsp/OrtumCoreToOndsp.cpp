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

    // An OrtumCore operation's own attributes are contract operands the
    // conversion consumes; the metadata guard is for every other op.
    if (isa_and_nonnull<ondrix::ortumcore::OrtumCoreDialect>(op->getDialect()))
      return WalkResult::advance();

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

template <typename SourceOp, typename DestinationOp,
          ondrix::ortumcore::ProductDomain (*getProductDomain)()>
class MacLikeOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!isa<ondrix::ondsp::AccType>(adaptor.getAcc().getType()) ||
        !isa<ondrix::ondsp::AccType>(resultType))
      return op.emitOpError("emulation requires converted Ondsp accumulator types");

    ondrix::ortumcore::ProductDomain domain = getProductDomain();
    auto numeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), domain.signedness,
        IntegerType::get(rewriter.getContext(), domain.operandWidth), domain.operandFrac);
    auto product = ondrix::ondsp::ProductAttr::get(rewriter.getContext(), domain.product.selection);
    rewriter.replaceOpWithNewOp<DestinationOp>(op, resultType, adaptor.getAcc(), adaptor.getLhs(),
                                               adaptor.getRhs(), numeric, product);
    return success();
  }
};

using MacAddOpLowering = MacLikeOpLowering<ondrix::ortumcore::MacAddOp, ondrix::ondsp::MacOp,
                                           ondrix::ortumcore::getSignedQ15FullProductDomain>;
using MacSubOpLowering = MacLikeOpLowering<ondrix::ortumcore::MacSubOp, ondrix::ondsp::MacSubOp,
                                           ondrix::ortumcore::getSignedQ15FullProductDomain>;
// The Q31 family differs only in its product domain: the raw high half lands
// at the accumulator's own fractional position, so the update, the state, and
// the readout are the Q15 family's unchanged.
using Q31MacAddOpLowering = MacLikeOpLowering<ondrix::ortumcore::Q31MacAddOp, ondrix::ondsp::MacOp,
                                              ondrix::ortumcore::getSignedQ31RawHighProductDomain>;
using Q31MacSubOpLowering =
    MacLikeOpLowering<ondrix::ortumcore::Q31MacSubOp, ondrix::ondsp::MacSubOp,
                      ondrix::ortumcore::getSignedQ31RawHighProductDomain>;

// The scaled saturating add/sub is exactly one declared ondsp shift boundary
// over the exact sum: same widening, same floor, same saturating narrowing.
template <typename SourceOp, typename DestinationOp>
class ScaledBinaryOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ondrix::ortumcore::ScaledBinaryDomain domain =
        ondrix::ortumcore::getShiftedSaturatingI32ScaledBinaryDomain();
    if (op.getShift() < 0 || uint64_t(op.getShift()) > domain.maxShift)
      return op.emitOpError("emulation requires a shift inside the target's scaling range");
    auto storage = IntegerType::get(rewriter.getContext(), domain.storageWidth);
    auto scale = ondrix::ondsp::ScaleAttr::get(rewriter.getContext(), /*preShiftLeft=*/0,
                                               unsigned(op.getShift()), domain.rounding,
                                               domain.overflow, storage);
    rewriter.replaceOpWithNewOp<DestinationOp>(op, storage, adaptor.getLhs(), adaptor.getRhs(),
                                               scale);
    return success();
  }
};

using SatShiftAddOpLowering =
    ScaledBinaryOpLowering<ondrix::ortumcore::SatShiftAddOp, ondrix::ondsp::AddShiftOp>;
using SatShiftSubOpLowering =
    ScaledBinaryOpLowering<ondrix::ortumcore::SatShiftSubOp, ondrix::ondsp::SubShiftOp>;

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

// Packed-complex emulation helpers: extract signed Q15 halves, compute
// exactly wide, and narrow every component through ondsp.round_shift so the
// packed pipeline inherits the proven rounding machinery.
static Value extractComponent(OpBuilder &builder, Location loc, Value packed, bool high,
                              Type carrier) {
  Value component = packed;
  if (high) {
    Value sixteen = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(16));
    component = builder.create<arith::ShRSIOp>(loc, packed, sixteen);
  }
  Value half = builder.create<arith::TruncIOp>(loc, builder.getI16Type(), component);
  return builder.create<arith::ExtSIOp>(loc, carrier, half);
}

static Value narrowComponent(OpBuilder &builder, Location loc, Value wide, int64_t shift,
                             ondrix::ortumcore::CxRounding rounding,
                             ondrix::ortumcore::CxOverflow overflow) {
  auto mode = rounding == ondrix::ortumcore::CxRounding::TowardNegative
                  ? ondrix::ondsp::RoundingMode::TowardNegative
                  : ondrix::ondsp::RoundingMode::NearestTiesPositive;
  auto narrowing = overflow == ondrix::ortumcore::CxOverflow::Wrap
                       ? ondrix::ondsp::OverflowMode::Wrap
                       : ondrix::ondsp::OverflowMode::Saturate;
  auto scale = ondrix::ondsp::ScaleAttr::get(builder.getContext(), 0, unsigned(shift), mode,
                                             narrowing, builder.getI16Type());
  return builder.create<ondrix::ondsp::RoundShiftOp>(loc, builder.getI16Type(), wide, scale);
}

static Value packComponents(OpBuilder &builder, Location loc, Value hiComponent,
                            Value loComponent) {
  Value hi = builder.create<arith::ExtSIOp>(loc, builder.getI32Type(), hiComponent);
  Value lo = builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), loComponent);
  Value sixteen = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(16));
  Value shifted = builder.create<arith::ShLIOp>(loc, hi, sixteen);
  return builder.create<arith::OrIOp>(loc, shifted, lo);
}

class CxMulConjOpLowering final : public OpConversionPattern<ondrix::ortumcore::CxMulConjOp> {
public:
  using OpConversionPattern<ondrix::ortumcore::CxMulConjOp>::OpConversionPattern;

  // The imaginary cross sum reaches +2^31 at value = twiddle = (-1, -1), so
  // the exact products live in an i64 carrier.
  LogicalResult matchAndRewrite(ondrix::ortumcore::CxMulConjOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type carrier = rewriter.getI64Type();
    Value vr = extractComponent(rewriter, loc, adaptor.getValue(), false, carrier);
    Value vi = extractComponent(rewriter, loc, adaptor.getValue(), true, carrier);
    Value wr = extractComponent(rewriter, loc, adaptor.getTwiddle(), false, carrier);
    Value wi = extractComponent(rewriter, loc, adaptor.getTwiddle(), true, carrier);
    Value real = rewriter.create<arith::AddIOp>(loc, rewriter.create<arith::MulIOp>(loc, vr, wr),
                                                rewriter.create<arith::MulIOp>(loc, vi, wi));
    Value imag = rewriter.create<arith::SubIOp>(loc, rewriter.create<arith::MulIOp>(loc, vi, wr),
                                                rewriter.create<arith::MulIOp>(loc, vr, wi));
    Value realNarrow =
        narrowComponent(rewriter, loc, real, op.getShift(), op.getRounding(), op.getOverflow());
    Value imagNarrow =
        narrowComponent(rewriter, loc, imag, op.getShift(), op.getRounding(), op.getOverflow());
    bool imagHigh = op.getLayout() == ondrix::ortumcore::CxLayout::ImagHi;
    rewriter.replaceOp(op, packComponents(rewriter, loc, imagHigh ? imagNarrow : realNarrow,
                                          imagHigh ? realNarrow : imagNarrow));
    return success();
  }
};

class CxBflyOpLowering final : public OpConversionPattern<ondrix::ortumcore::CxBflyOp> {
public:
  using OpConversionPattern<ondrix::ortumcore::CxBflyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ortumcore::CxBflyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type carrier = rewriter.getI32Type();
    Value ar = extractComponent(rewriter, loc, adaptor.getLhs(), false, carrier);
    Value ai = extractComponent(rewriter, loc, adaptor.getLhs(), true, carrier);
    Value br = extractComponent(rewriter, loc, adaptor.getRhs(), false, carrier);
    Value bi = extractComponent(rewriter, loc, adaptor.getRhs(), true, carrier);
    Value sumReal = rewriter.create<arith::AddIOp>(loc, ar, br);
    Value diffReal = rewriter.create<arith::SubIOp>(loc, ar, br);
    Value sumImag = rewriter.create<arith::AddIOp>(loc, ai, bi);
    Value diffImag = rewriter.create<arith::SubIOp>(loc, ai, bi);
    // Both variants keep sum-of-reals in out0 and difference-of-reals in
    // out1; cross swaps only the imaginary halves.
    bool cross = op.getVariant() == ondrix::ortumcore::CxBflyVariant::Cross;
    auto narrow = [&](Value component) {
      return narrowComponent(rewriter, loc, component, op.getShift(), op.getRounding(),
                             op.getOverflow());
    };
    Value out0 = packComponents(rewriter, loc, narrow(cross ? diffImag : sumImag), narrow(sumReal));
    Value out1 =
        packComponents(rewriter, loc, narrow(cross ? sumImag : diffImag), narrow(diffReal));
    rewriter.replaceOp(op, {out0, out1});
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
    patterns.add<AccInitOpLowering, MacAddOpLowering, MacSubOpLowering, Q31MacAddOpLowering,
                 Q31MacSubOpLowering, DmacOpLowering, AccOutOpLowering, SatShiftAddOpLowering,
                 SatShiftSubOpLowering, CxMulConjOpLowering, CxBflyOpLowering, BitrevAddOpLowering,
                 BitrevSubOpLowering>(typeConverter, &getContext());
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
