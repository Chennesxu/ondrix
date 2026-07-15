#include "ondrix/Conversion/OndspToScalar/OndspToScalar.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Support/FixedPointSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/AttrTypeSubElements.h"
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

static bool isSignedQ15(ondrix::ondsp::FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 16 && numeric.getFrac() == 15 &&
         numeric.getSignedness() == ondrix::ondsp::Signedness::Signed;
}

static bool isFullProduct(ondrix::ondsp::ProductAttr product) {
  return product.getSelection() == ondrix::ondsp::ProductSelection::Full;
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

static bool isNestedAccumulatorContainer(Type type) {
  return !isa<ondrix::ondsp::AccType>(type) && containsOndspAccumulator(type);
}

static LogicalResult verifyNoNestedAccumulatorContainers(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    auto containsNested = [](TypeRange types) {
      return llvm::any_of(types, isNestedAccumulatorContainer);
    };
    if (containsNested(op->getOperandTypes()) || containsNested(op->getResultTypes())) {
      op->emitOpError("nested accumulator containers are unsupported");
      return WalkResult::interrupt();
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      FunctionType type = function.getFunctionType();
      if (containsNested(type.getInputs()) || containsNested(type.getResults())) {
        op->emitOpError("nested accumulator containers are unsupported");
        return WalkResult::interrupt();
      }
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (containsNested(block.getArgumentTypes())) {
          op->emitOpError("nested accumulator containers are unsupported");
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

class AccImportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccImportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccImportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccImportOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!isSupportedQ15Accumulator(accumulator) || !isSignedQ15(op.getSrc()))
      return op.emitOpError(
          "Q15 scalar lowering requires signed i16 frac=15 input and signed i40 frac=30 "
          "accumulator");

    auto accumulatorStorage = cast<IntegerType>(accumulator.getStorage());
    Value extended =
        rewriter.create<arith::ExtSIOp>(op.getLoc(), accumulatorStorage, adaptor.getInput());
    Value shift = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), accumulator.getFrac() - op.getSrc().getFrac(), accumulatorStorage.getWidth());
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, extended, shift);
    return success();
  }
};

static Value lowerAccumulatorUpdate(Location loc, Value accumulator, Value product,
                                    ondrix::ondsp::OverflowMode overflowMode,
                                    ondrix::fixedpoint::AccumulatorUpdateOperation operation,
                                    OpBuilder &builder) {
  auto accumulatorType = cast<IntegerType>(accumulator.getType());
  auto productType = cast<IntegerType>(product.getType());
  unsigned intermediateWidth = ondrix::fixedpoint::getAccumulatorUpdateIntermediateWidth(
      accumulatorType.getWidth(), productType.getWidth());
  IntegerType intermediateType = builder.getIntegerType(intermediateWidth);
  Value extendedAccumulator = builder.create<arith::ExtSIOp>(loc, intermediateType, accumulator);
  Value extendedProduct = builder.create<arith::ExtSIOp>(loc, intermediateType, product);
  Value updated;
  switch (operation) {
  case ondrix::fixedpoint::AccumulatorUpdateOperation::Add:
    updated = builder.create<arith::AddIOp>(loc, extendedAccumulator, extendedProduct);
    break;
  case ondrix::fixedpoint::AccumulatorUpdateOperation::Subtract:
    updated = builder.create<arith::SubIOp>(loc, extendedAccumulator, extendedProduct);
    break;
  }

  if (overflowMode == ondrix::ondsp::OverflowMode::Wrap)
    return builder.create<arith::TruncIOp>(loc, accumulatorType, updated);

  // Clamp in the exact update width before narrowing to the accumulator.
  llvm::APInt minimum =
      llvm::APInt::getSignedMinValue(accumulatorType.getWidth()).sext(intermediateWidth);
  llvm::APInt maximum =
      llvm::APInt::getSignedMaxValue(accumulatorType.getWidth()).sext(intermediateWidth);
  Value minimumValue = builder.create<arith::ConstantOp>(
      loc, intermediateType, builder.getIntegerAttr(intermediateType, minimum));
  Value maximumValue = builder.create<arith::ConstantOp>(
      loc, intermediateType, builder.getIntegerAttr(intermediateType, maximum));
  Value belowMinimum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, updated, minimumValue);
  Value aboveMaximum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, updated, maximumValue);
  Value lowerClamped = builder.create<arith::SelectOp>(loc, belowMinimum, minimumValue, updated);
  Value clamped = builder.create<arith::SelectOp>(loc, aboveMaximum, maximumValue, lowerClamped);
  return builder.create<arith::TruncIOp>(loc, accumulatorType, clamped);
}

static Value createIntegerConstant(Location loc, IntegerType type, int64_t value,
                                   ConversionPatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantOp>(loc, type, rewriter.getIntegerAttr(type, value));
}

static Value roundSignedRightShift(Location loc, Value input, unsigned shift,
                                   ondrix::ondsp::RoundingMode roundingMode,
                                   ConversionPatternRewriter &rewriter) {
  auto type = cast<IntegerType>(input.getType());
  if (shift == 0)
    return input;

  Value shiftValue = createIntegerConstant(loc, type, shift, rewriter);
  Value quotient = rewriter.create<arith::ShRSIOp>(loc, input, shiftValue);
  if (roundingMode == ondrix::ondsp::RoundingMode::TowardNegative)
    return quotient;

  IntegerType remainderBitsType = rewriter.getIntegerType(shift);
  Value remainderBits = rewriter.create<arith::TruncIOp>(loc, remainderBitsType, input);
  Value remainder = rewriter.create<arith::ExtUIOp>(loc, type, remainderBits);
  Value zero = createIntegerConstant(loc, type, 0, rewriter);
  Value one = createIntegerConstant(loc, type, 1, rewriter);
  Value incrementCondition;

  switch (roundingMode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    llvm_unreachable("toward-negative rounding returned above");
  case ondrix::ondsp::RoundingMode::TowardZero: {
    Value isNegative = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, zero);
    Value hasRemainder =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero);
    incrementCondition = rewriter.create<arith::AndIOp>(loc, isNegative, hasRemainder);
    break;
  }
  case ondrix::ondsp::RoundingMode::NearestEven: {
    Value half = createIntegerConstant(loc, type, int64_t{1} << (shift - 1), rewriter);
    Value aboveHalf =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remainder, half);
    Value exactlyHalf =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, half);
    Value quotientLowBit = rewriter.create<arith::AndIOp>(loc, quotient, one);
    Value quotientIsOdd =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotientLowBit, zero);
    Value halfAndOdd = rewriter.create<arith::AndIOp>(loc, exactlyHalf, quotientIsOdd);
    incrementCondition = rewriter.create<arith::OrIOp>(loc, aboveHalf, halfAndOdd);
    break;
  }
  }

  Value increment = rewriter.create<arith::SelectOp>(loc, incrementCondition, one, zero);
  return rewriter.create<arith::AddIOp>(loc, quotient, increment);
}

static Value narrowSignedValue(Location loc, Value input, IntegerType destinationType,
                               ondrix::ondsp::OverflowMode overflowMode,
                               ConversionPatternRewriter &rewriter) {
  auto inputType = cast<IntegerType>(input.getType());
  if (overflowMode == ondrix::ondsp::OverflowMode::Wrap)
    return rewriter.create<arith::TruncIOp>(loc, destinationType, input);

  llvm::APInt minimum =
      llvm::APInt::getSignedMinValue(destinationType.getWidth()).sext(inputType.getWidth());
  llvm::APInt maximum =
      llvm::APInt::getSignedMaxValue(destinationType.getWidth()).sext(inputType.getWidth());
  Value minimumValue = rewriter.create<arith::ConstantOp>(
      loc, inputType, rewriter.getIntegerAttr(inputType, minimum));
  Value maximumValue = rewriter.create<arith::ConstantOp>(
      loc, inputType, rewriter.getIntegerAttr(inputType, maximum));
  Value belowMinimum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, minimumValue);
  Value aboveMaximum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, input, maximumValue);
  Value lowerClamped = rewriter.create<arith::SelectOp>(loc, belowMinimum, minimumValue, input);
  Value clamped = rewriter.create<arith::SelectOp>(loc, aboveMaximum, maximumValue, lowerClamped);
  return rewriter.create<arith::TruncIOp>(loc, destinationType, clamped);
}

template <typename OpTy, ondrix::fixedpoint::AccumulatorUpdateOperation operation>
class MacLikeOpLowering final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!isSupportedQ15Accumulator(accumulator) || !isSignedQ15(op.getNumeric()) ||
        !isFullProduct(op.getProduct()))
      return op.emitOpError(
          "Q15 scalar lowering requires signed i16 frac=15 full product and signed i40 "
          "frac=30 accumulator");

    IntegerType productType = rewriter.getI32Type();
    Value lhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productType, adaptor.getLhs());
    Value rhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productType, adaptor.getRhs());
    Value product = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, rhs);
    Value updated = lowerAccumulatorUpdate(op.getLoc(), adaptor.getAcc(), product,
                                           accumulator.getUpdateOverflow(), operation, rewriter);
    rewriter.replaceOp(op, updated);
    return success();
  }
};

using MacOpLowering =
    MacLikeOpLowering<ondrix::ondsp::MacOp, ondrix::fixedpoint::AccumulatorUpdateOperation::Add>;
using MacSubOpLowering =
    MacLikeOpLowering<ondrix::ondsp::MacSubOp,
                      ondrix::fixedpoint::AccumulatorUpdateOperation::Subtract>;

static FailureOr<MemRefType> getRankOneQ15MemRefType(Operation *op, Value value,
                                                     StringRef operandName) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 1 || !type.getElementType().isSignlessInteger(16))
    return op->emitOpError() << operandName
                             << " must be a rank-1 memref<Nxi16> for Q15 scalar lowering";
  return type;
}

static Value getDimZeroSize(Location loc, Value value, MemRefType type, Value zeroIndex,
                            ConversionPatternRewriter &rewriter) {
  if (!type.isDynamicDim(0))
    return rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(0));
  return rewriter.create<memref::DimOp>(loc, value, zeroIndex);
}

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
    auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (!accumulator || !numeric || !op.getProduct() || !isSupportedQ15Accumulator(accumulator) ||
        !isSignedQ15(numeric) || !isFullProduct(*op.getProduct()))
      return op.emitOpError(
          "Q15 scalar reduction requires signed i16 frac=15 full product and signed i40 "
          "frac=30 accumulator");

    FailureOr<MemRefType> lhsType = getRankOneQ15MemRefType(op, op.getLhs(), "lhs");
    if (failed(lhsType))
      return failure();
    FailureOr<MemRefType> rhsType = getRankOneQ15MemRefType(op, op.getRhs(), "rhs");
    if (failed(rhsType))
      return failure();

    Location loc = op.getLoc();
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = getDimZeroSize(loc, adaptor.getLhs(), *lhsType, lowerBound, rewriter);
    if (lhsType->isDynamicDim(0) || rhsType->isDynamicDim(0)) {
      Value rhsSize = getDimZeroSize(loc, adaptor.getRhs(), *rhsType, lowerBound, rewriter);
      Value lengthsMatch =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, upperBound, rhsSize);
      rewriter.create<cf::AssertOp>(
          loc, lengthsMatch,
          rewriter.getStringAttr("ondsp.reduce_mac requires equal operand lengths"));
    }
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{adaptor.getInitial()},
        [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), iv);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), iv);
          IntegerType productType = builder.getI32Type();
          Value lhsExtended = builder.create<arith::ExtSIOp>(bodyLoc, productType, lhs);
          Value rhsExtended = builder.create<arith::ExtSIOp>(bodyLoc, productType, rhs);
          Value product = builder.create<arith::MulIOp>(bodyLoc, lhsExtended, rhsExtended);
          Value next = lowerAccumulatorUpdate(
              bodyLoc, iterArgs.front(), product, accumulator.getUpdateOverflow(),
              ondrix::fixedpoint::AccumulatorUpdateOperation::Add, builder);
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class AccExportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccExportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccExportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExportOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!isSupportedQ15Accumulator(accumulator) || !isSignedQ15(op.getDst()) ||
        !op.getResult().getType().isSignlessInteger(16))
      return op.emitOpError(
          "Q15 scalar lowering requires signed i40 frac=30 accumulator and signed i16 "
          "frac=15 destination");

    unsigned shift = accumulator.getFrac() - op.getDst().getFrac();
    Value rounded =
        roundSignedRightShift(op.getLoc(), adaptor.getAcc(), shift, op.getRounding(), rewriter);
    Value result =
        narrowSignedValue(op.getLoc(), rounded, rewriter.getI16Type(), op.getOverflow(), rewriter);
    rewriter.replaceOp(op, result);
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
    if (llvm::any_of(op->getAttrs(), [](NamedAttribute namedAttribute) {
          return containsOndspAccumulator(namedAttribute.getValue());
        }))
      return op.emitOpError("cannot convert attributes containing source accumulator types");

    auto replacement =
        rewriter.create<arith::SelectOp>(op.getLoc(), resultType, adaptor.getCondition(),
                                         adaptor.getTrueValue(), adaptor.getFalseValue());
    // This is a structural type conversion of the same operation, so preserve
    // its verified semantic and discardable attributes.
    replacement->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class ConvertOndspQ15ToScalarPass final
    : public ondrix::impl::ConvertOndspQ15ToScalarBase<ConvertOndspQ15ToScalarPass> {
public:
  using ondrix::impl::ConvertOndspQ15ToScalarBase<
      ConvertOndspQ15ToScalarPass>::ConvertOndspQ15ToScalarBase;

  void runOnOperation() override {
    if (failed(verifyNoNestedAccumulatorContainers(getOperation()))) {
      signalPassFailure();
      return;
    }

    OndspQ15ToScalarTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    patterns.add<AccExportOpLowering, AccImportOpLowering, AccZeroOpLowering, MacOpLowering,
                 MacSubOpLowering, ReduceMacOpLowering, SelectOpTypeConversion>(typeConverter,
                                                                                &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<ondrix::ondsp::OndspDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
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
