#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"
#include "ondrix/Conversion/Utils/ConversionLegality.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"
#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/AttrTypeSubElements.h"
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

static ondrix::ortumcore::AccumulatorDomain
getAccumulatorDomain(ondrix::ondsp::AccType accumulator) {
  return {cast<IntegerType>(accumulator.getStorage()).getWidth(), accumulator.getFrac(),
          accumulator.getSignedness(), accumulator.getUpdateOverflow()};
}

static FailureOr<ondrix::ortumcore::ProductDomain>
getProductDomain(Operation *op, ondrix::ondsp::FixedAttr numeric,
                 ondrix::ondsp::ProductAttr product) {
  FailureOr<ondrix::ondsp::ProductSemantics> semantics =
      ondrix::ondsp::inferProductSemantics(op, numeric, product);
  if (failed(semantics))
    return failure();
  return ondrix::ortumcore::ProductDomain{cast<IntegerType>(numeric.getStorage()).getWidth(),
                                          numeric.getFrac(), numeric.getSignedness(), *semantics};
}

// The parameterless target type admits only this accumulator representation.
// Product semantics remain operation-specific legalization rules.
static bool isSupportedOrtumCoreAccumulator(ondrix::ondsp::AccType accumulator) {
  // The target accumulator domain has no lane count, so a lane count greater
  // than one would be silently dropped on the way to the parameterless target
  // type. Refuse it before the domain comparison rather than after.
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return false;
  return ondrix::ortumcore::OrtumCoreTargetProfile().supportsAccumulator(
      getAccumulatorDomain(accumulator));
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

static bool isOndspType(Type type) {
  return type.getDialect().getNamespace() == ondrix::ondsp::OndspDialect::getDialectNamespace();
}

static bool isOndspAttribute(Attribute attribute) {
  return attribute.getDialect().getNamespace() ==
         ondrix::ondsp::OndspDialect::getDialectNamespace();
}

static LogicalResult verifyOrtumCoreAccumulator(Operation *op, ondrix::ondsp::AccType accumulator) {
  if (!isSupportedOrtumCoreAccumulator(accumulator))
    return op->emitOpError(
        "ortumcore lowering requires a signed 40-bit ondsp accumulator with frac=30 and "
        "saturating updates");
  return success();
}

static LogicalResult verifySupportedMacPolicy(Operation *op, ondrix::ondsp::AccType accumulator,
                                              ondrix::ondsp::FixedAttr numeric,
                                              ondrix::ondsp::ProductAttr product) {
  FailureOr<ondrix::ortumcore::ProductDomain> productDomain =
      getProductDomain(op, numeric, product);
  if (failed(productDomain))
    return failure();
  if (ondrix::ortumcore::OrtumCoreTargetProfile().supportsMac(*productDomain,
                                                              getAccumulatorDomain(accumulator)))
    return success();

  if (ondrix::ondsp::isSignedQ31(numeric) && ondrix::ondsp::isRawHighProduct(product))
    return op->emitOpError(
        "q31 raw-high target equivalence is not specified; lower through a proven scalar "
        "sequence first");

  return op->emitOpError("ortumcore MAC lowering supports only signed q15 full-product semantics");
}

static bool containsOndspAccumulator(Type type) {
  // TypeConverter legality is shallow for aggregate types; reject source
  // accumulators recursively so tuples and other wrappers cannot leak through.
  return ondrix::conversion::containsMatchingType(
      type, [](Type nested) { return isa<ondrix::ondsp::AccType>(nested); });
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
    SmallVector<Type> types;
    ondrix::ondsp::appendAccumulatorCandidateTypes(op, types);

    for (Type type : types) {
      ondrix::ondsp::AccType unsupported =
          ondrix::ondsp::findRejectedAccumulator(type, isSupportedOrtumCoreAccumulator);
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
  return ondrix::conversion::hasLegalConvertedTypesAndAttributes(op, typeConverter, isOndspType,
                                                                 isOndspAttribute);
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

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExportOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ondrix::ondsp::AccType accumulator = op.getAcc().getType();
    ondrix::ondsp::FixedAttr destination = op.getDst();
    auto storage = dyn_cast<IntegerType>(destination.getStorage());
    int64_t shift = int64_t(accumulator.getFrac()) - int64_t(destination.getFrac());
    if (!storage || !isa<IntegerType>(op.getResult().getType()) ||
        !ondrix::ortumcore::OrtumCoreTargetProfile().supportsExport(
            getAccumulatorDomain(accumulator), op.getRounding(), op.getOverflow(), shift) ||
        (storage.getWidth() != 32 && storage.getWidth() != 16))
      return op.emitOpError(
          "accumulator export is outside the proven readout capability (toward_negative "
          "rounding, saturating i32 or i16 destination, fractional shift in [0, 15])");

    if (!isa<ondrix::ortumcore::AccumType>(adaptor.getAcc().getType()))
      return op.emitOpError("export lowering requires a converted target accumulator operand");

    Location loc = op.getLoc();
    Value out = rewriter.create<ondrix::ortumcore::AccOutOp>(loc, rewriter.getI32Type(),
                                                             adaptor.getAcc(), shift);
    if (storage.getWidth() == 32) {
      rewriter.replaceOp(op, out);
      return success();
    }
    // The i16 destination clamp composes exactly: the readout's wider i32
    // saturation cannot change a subsequent narrower clamp (Passes.td carries
    // the argument).
    Value minimum = rewriter.create<arith::ConstantIntOp>(loc, -32768, 32);
    Value maximum = rewriter.create<arith::ConstantIntOp>(loc, 32767, 32);
    Value belowMinimum =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, out, minimum);
    Value lowerClamped = rewriter.create<arith::SelectOp>(loc, belowMinimum, minimum, out);
    Value aboveMaximum =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lowerClamped, maximum);
    Value clamped = rewriter.create<arith::SelectOp>(loc, aboveMaximum, maximum, lowerClamped);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, rewriter.getI16Type(), clamped);
    return success();
  }
};

class MacOpLowering final : public OpConversionPattern<ondrix::ondsp::MacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::MacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::MacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyOrtumCoreAccumulator(op, op.getAcc().getType())) ||
        failed(
            verifySupportedMacPolicy(op, op.getAcc().getType(), op.getNumeric(), op.getProduct())))
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
        failed(
            verifySupportedMacPolicy(op, op.getAcc().getType(), op.getNumeric(), op.getProduct())))
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
                 MacSubOpLowering, ReduceMacOpLowering>(typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<ondrix::ortumcore::OrtumCoreDialect>(
        [&](Operation *op) { return hasLegalConvertedTypes(op, typeConverter); });
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
    // Accumulator values flow through structured control flow in whole
    // kernels; convert scf region signatures with the same type converter.
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspToOrtumCorePass() {
  return std::make_unique<ConvertOndspToOrtumCorePass>();
}
