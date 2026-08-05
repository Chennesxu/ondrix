#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Support/DSPTypeUtils.h"
#include "ondrix/Support/Q31TwiddleTables.h"

#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace ondrix::ir;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixOps.cpp.inc"

namespace {

static void addMemRefReadEffect(Value value,
                                SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<BaseMemRefType>(value.getType()))
    return;
  effects.emplace_back(MemoryEffects::Read::get(), value, SideEffects::DefaultResource::get());
}

static LogicalResult verifyValueOnlyTypes(Operation *op) {
  auto containsMemRef = [](Type type) {
    return type.walk([](BaseMemRefType) { return WalkResult::interrupt(); }).wasInterrupted();
  };
  if (llvm::any_of(op->getOperandTypes(), containsMemRef))
    return op->emitOpError("value-only operation does not accept memref operands");
  if (llvm::any_of(op->getResultTypes(), containsMemRef))
    return op->emitOpError("value-only operation does not produce memref results");
  if (llvm::any_of(op->getOperandTypes(), ondrix::containsScalableVectorType))
    return op->emitOpError("value-only operation does not accept scalable vector operands");
  if (llvm::any_of(op->getResultTypes(), ondrix::containsScalableVectorType))
    return op->emitOpError("value-only operation does not produce scalable vector results");
  if (llvm::any_of(op->getOperandTypes(), ondrix::containsDynamicOrUnrankedShapedType))
    return op->emitOpError(
        "value-only operation does not accept dynamic or unranked shaped operands");
  if (llvm::any_of(op->getResultTypes(), ondrix::containsDynamicOrUnrankedShapedType))
    return op->emitOpError(
        "value-only operation does not produce dynamic or unranked shaped results");
  return success();
}

static Type getNumericStorage(Attribute numeric) {
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(numeric))
    return fixed.getStorage();
  return cast<ondrix::ondsp::FpAttr>(numeric).getFormat();
}

static LogicalResult verifyFixedReductionResult(Operation *op, Type resultType,
                                                ondrix::ondsp::FixedAttr numeric,
                                                ondrix::ondsp::ProductAttr product) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(resultType);
  if (!accumulator)
    return op->emitOpError("fixed reduction result must use !ondsp.acc");
  // Algorithm intent never declares a lane count: batching across outputs is a
  // lowering decision an explicit pass makes, so a multi-lane accumulator in
  // the intent attribute would be an unproven claim about the schedule.
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return op->emitOpError("algorithm accumulator must be single-lane; lane batching is a lowering "
                           "decision, not part of the algorithm contract");
  if (accumulator.getSignedness() != numeric.getSignedness())
    return op->emitOpError("accumulator signedness must match fixed numeric policy");
  FailureOr<ondrix::ondsp::ProductSemantics> semantics =
      ondrix::ondsp::inferProductSemantics(op, numeric, product);
  if (failed(semantics))
    return failure();
  if (accumulator.getFrac() != semantics->frac)
    return op->emitOpError() << "accumulator frac " << accumulator.getFrac()
                             << " does not match product frac " << semantics->frac;
  return success();
}

static LogicalResult verifyButterflyValueDomain(ButterflyOp op) {
  if (op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return op.emitOpError("executable butterfly requires packed_i16_imag_hi_real_lo layout");
  if (!llvm::all_of(op.getOperandTypes(), [](Type type) { return type.isSignlessInteger(32); }) ||
      !llvm::all_of(op.getResultTypes(), [](Type type) { return type.isSignlessInteger(32); }))
    return op.emitOpError("executable butterfly requires scalar signless i32 packed values");
  return success();
}

static LogicalResult verifyUnencodedTensorTypes(Operation *op,
                                                ArrayRef<RankedTensorType> tensorTypes) {
  if (llvm::any_of(tensorTypes, [](RankedTensorType type) { return type.getEncoding(); }))
    return op->emitOpError("does not support encoded tensor types");
  return success();
}

static LogicalResult verifyCfftValueDomain(CfftOp op) {
  // The layout selects the executable profile: packed Q15 in an i32 container
  // or packed Q31 in an i64 container. The Q15 extent bound is the
  // in-compiler twiddle contract; the Q31 bound is the frozen offline table,
  // so an extent beyond it fails closed rather than extrapolating.
  std::optional<ondrix::ondsp::PackedComplexProfile> profile =
      ondrix::ondsp::getPackedComplexProfile(op.getLayout().getLayout());
  if (!profile)
    return op.emitOpError("executable CFFT requires packed_i16_imag_hi_real_lo or "
                          "packed_i32_imag_hi_real_lo layout");
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType resultType = op.getResult().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t maximumExtent = profile->storageWidth == 16 ? 1024 : ondrix::kMaxQ31TwiddleExtent;
  unsigned containerWidth = profile->containerWidth;
  int64_t extent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  if (inputType != resultType || inputType.getRank() != 1 || extent < 4 || extent > maximumExtent ||
      !llvm::isPowerOf2_64(extent) || !inputType.getElementType().isSignlessInteger(containerWidth))
    return op.emitOpError() << "executable CFFT requires matching tensor<Nxi" << containerWidth
                            << "> input and result with power-of-two N in [4, " << maximumExtent
                            << "]";
  return success();
}

static LogicalResult verifyRfftValueDomain(RfftOp op) {
  if (op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return op.emitOpError("executable RFFT requires packed_i16_imag_hi_real_lo layout");
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType resultType = op.getResult().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent < 8 || inputExtent > 1024 || !llvm::isPowerOf2_64(inputExtent) ||
      resultExtent != inputExtent / 2 + 1 || !inputType.getElementType().isSignlessInteger(16) ||
      !resultType.getElementType().isSignlessInteger(32))
    return op.emitOpError("executable RFFT requires tensor<Nxi16> to tensor<(N/2+1)xi32> "
                          "with power-of-two N in [8, 1024]");
  return success();
}

static LogicalResult verifyIrfftValueDomain(IrfftOp op) {
  if (op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return op.emitOpError("executable IRFFT requires packed_i16_imag_hi_real_lo layout");
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType resultType = op.getResult().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (resultExtent < 8 || resultExtent > 1024 || !llvm::isPowerOf2_64(resultExtent) ||
      inputExtent != resultExtent / 2 + 1 || !inputType.getElementType().isSignlessInteger(32) ||
      !resultType.getElementType().isSignlessInteger(16))
    return op.emitOpError("executable IRFFT requires tensor<(N/2+1)xi32> to tensor<Nxi16> "
                          "with power-of-two N in [8, 1024]");
  return success();
}

static LogicalResult verifySignedFixedFormat(Operation *op, Attribute numeric, unsigned width,
                                             unsigned frac, StringRef name) {
  auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(numeric);
  if (!fixed || fixed.getSignedness() != ondrix::ondsp::Signedness::Signed ||
      !fixed.getStorage().isSignlessInteger(width) || fixed.getFrac() != frac)
    return op->emitOpError() << name << " requires #ondsp.fixed<signed, storage = i" << width
                             << ", frac = " << frac << ">";
  return success();
}

// Rounding is a per-operation declared contract, never a mode that a new
// dialect enum case silently extends. Operations that already pin their own
// admissible tie rules keep doing so; this helper is the shared floor for
// the remaining rounding-bearing operations, whose contracts, lowerings, and
// differential evidence cover only the three established modes. A newly
// declared mode has to be admitted deliberately, per operation, together
// with its lowering and evidence.
static LogicalResult verifyEstablishedRounding(Operation *op, ondrix::ondsp::RoundingMode rounding,
                                               StringRef name) {
  switch (rounding) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
  case ondrix::ondsp::RoundingMode::TowardZero:
  case ondrix::ondsp::RoundingMode::NearestEven:
    return success();
  case ondrix::ondsp::RoundingMode::NearestTiesPositive:
    break;
  }
  return op->emitOpError() << name
                           << " supports toward_negative, toward_zero, or nearest_even rounding";
}

static LogicalResult verifyRfftRadix4SplitValueDomain(RfftRadix4SplitOp op) {
  if (op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return op.emitOpError(
        "executable radix-4 split RFFT requires packed_i16_imag_hi_real_lo layout");
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType resultType = op.getResult().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent != 32 || resultExtent != 17 ||
      !inputType.getElementType().isSignlessInteger(16) ||
      !resultType.getElementType().isSignlessInteger(32))
    return op.emitOpError(
        "executable radix-4 split RFFT requires tensor<32xi16> to tensor<17xi32>");
  return success();
}

static LogicalResult verifyDesignCoefficientTensor(Operation *op, RankedTensorType type,
                                                   int64_t minExtent, int64_t maxExtent) {
  if (failed(verifyUnencodedTensorTypes(op, {type})))
    return failure();
  if (type.getRank() != 1 || !type.hasStaticShape() || !type.getElementType().isSignlessInteger(16))
    return op->emitOpError("requires a static rank-1 i16 coefficient tensor");
  int64_t extent = type.getDimSize(0);
  if (extent < minExtent || extent > maxExtent)
    return op->emitOpError() << "coefficient extent must be in [" << minExtent << ", " << maxExtent
                             << "]";
  return success();
}

static LogicalResult verifyQuantizeDomain(QuantizeOp op) {
  if (!ondrix::haveSameElementwiseShape(op.getInput().getType(), op.getResult().getType()))
    return op.emitOpError("input and result must use the same scalar or static shaped domain");
  if (ondrix::getElementTypeOrSelf(op.getInput().getType()) != getNumericStorage(op.getSrc()))
    return op.emitOpError("input element type must match source numeric storage type");
  if (ondrix::getElementTypeOrSelf(op.getResult().getType()) != getNumericStorage(op.getDst()))
    return op.emitOpError("result element type must match destination numeric storage type");
  return success();
}

static LogicalResult verifyFirWindow(FirOp op) {
  if (isa<RankedTensorType>(op.getInput().getType()) ||
      isa<RankedTensorType>(op.getCoeffs().getType()))
    return op.emitOpError(
        "tensor FIR windows have no executable consumer; use memrefs or fixed vectors");

  auto inputType = dyn_cast<ShapedType>(op.getInput().getType());
  auto coeffType = dyn_cast<ShapedType>(op.getCoeffs().getType());
  if (!inputType || !coeffType || !inputType.hasRank() || !coeffType.hasRank() ||
      inputType.getRank() != 1 || coeffType.getRank() != 1)
    return op.emitOpError("requires rank-1 input and coefficient windows");
  if (ondrix::isScalableVectorType(op.getInput().getType()) ||
      ondrix::isScalableVectorType(op.getCoeffs().getType()))
    return op.emitOpError("scalable vector windows are not supported");
  if (isa<ondrix::ondsp::FpAttr>(op.getNumeric()) &&
      (isa<VectorType>(op.getInput().getType()) || isa<VectorType>(op.getCoeffs().getType())))
    return op.emitOpError("floating-point vector FIR windows have no executable consumer");

  if (inputType.getElementType() != coeffType.getElementType())
    return op.emitOpError("input and coefficient element types must match");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coeffLength = coeffType.getDimSize(0);
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coeffLength) &&
      inputLength != coeffLength)
    return op.emitOpError("input and coefficient windows must have equal length");

  Type elementType = inputType.getElementType();
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (elementType != fixed.getStorage())
      return op.emitOpError("window element type must match fixed numeric storage type");
    return verifyFixedReductionResult(op, op.getResult().getType(), fixed, *op.getProduct());
  }

  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "FIR window")))
    return failure();
  if (elementType != fp.getFormat() || op.getResult().getType() != fp.getFormat())
    return op.emitOpError("floating-point FIR window and result types must match numeric format");
  return success();
}

static LogicalResult verifyFirFilterDomain(FirFilterOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coeffType = op.getCoeffs().getType();
  RankedTensorType initType = op.getInit().getType();

  if (inputType.getRank() != 1 || coeffType.getRank() != 1 || initType.getRank() != 1)
    return op.emitOpError("requires rank-1 input, coefficient, and init tensors");
  if (inputType.getElementType() != coeffType.getElementType())
    return op.emitOpError("input and coefficient element types must match");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coefficientLength = coeffType.getDimSize(0);
  int64_t outputLength = initType.getDimSize(0);
  if (op.getOutputOrigin() && op.getBoundary() != FirBoundaryMode::Full)
    return op.emitOpError("output_origin is supported only for full FIR boundaries");
  if (!ShapedType::isDynamic(coefficientLength) && coefficientLength == 0)
    return op.emitOpError() << stringifyFirBoundaryMode(op.getBoundary())
                            << " FIR requires at least one coefficient";

  if (op.getBoundary() == FirBoundaryMode::Valid) {
    if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coefficientLength)) {
      if (inputLength < coefficientLength)
        return op.emitOpError("valid FIR input must cover one coefficient window");
      int64_t expectedOutputLength = inputLength - coefficientLength + 1;
      if (!ShapedType::isDynamic(outputLength) && outputLength != expectedOutputLength)
        return op.emitOpError() << "valid FIR output length must be " << expectedOutputLength;
    }
  } else {
    if (!ShapedType::isDynamic(inputLength) && inputLength == 0)
      return op.emitOpError("full FIR requires at least one input sample");
    if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coefficientLength)) {
      if (inputLength > std::numeric_limits<int64_t>::max() - coefficientLength + 1)
        return op.emitOpError("full FIR output length exceeds the indexable extent range");
      int64_t completeOutputLength = inputLength + coefficientLength - 1;
      if (!op.getOutputOrigin()) {
        if (!ShapedType::isDynamic(outputLength) && outputLength != completeOutputLength)
          return op.emitOpError() << "full FIR output length must be " << completeOutputLength;
      } else {
        APInt outputOriginValue;
        if (matchPattern(op.getOutputOrigin(), m_ConstantInt(&outputOriginValue))) {
          int64_t outputOrigin = outputOriginValue.getSExtValue();
          if (outputOrigin < 0 || outputOrigin > completeOutputLength)
            return op.emitOpError("full FIR output tile exceeds the complete output range");
          if (!ShapedType::isDynamic(outputLength) &&
              outputLength > completeOutputLength - outputOrigin)
            return op.emitOpError("full FIR output tile exceeds the complete output range");
        }
      }
    }
  }

  Type inputElement = inputType.getElementType();
  Type outputElement = initType.getElementType();
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (inputElement != fixed.getStorage())
      return op.emitOpError("input and coefficient element type must match fixed numeric storage");
    if (!op.getAccumulator() || !op.getDst() || !op.getRounding() || !op.getOverflow())
      return op.emitOpError(
          "fixed FIR filter requires accumulator, dst, rounding, and overflow attributes");
    if (failed(verifyEstablishedRounding(op, *op.getRounding(), "fixed FIR filter")))
      return failure();
    if (failed(verifyFixedReductionResult(op, *op.getAccumulator(), fixed, *op.getProduct())))
      return failure();
    if (op.getAccumulator()->getSignedness() != op.getDst()->getSignedness())
      return op.emitOpError("accumulator and destination signedness must match");
    if (op.getAccumulator()->getFrac() < op.getDst()->getFrac())
      return op.emitOpError("destination frac must not exceed accumulator frac");
    if (outputElement != op.getDst()->getStorage())
      return op.emitOpError("init and result element type must match destination storage");
    return success();
  }

  if (op.getAccumulator() || op.getDst() || op.getRounding() || op.getOverflow())
    return op.emitOpError(
        "floating-point FIR filter must not specify fixed-point accumulator or export policy");
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "FIR filter")))
    return failure();
  if (inputElement != fp.getFormat() || outputElement != fp.getFormat())
    return op.emitOpError("floating-point input, coefficients, init, and result must match format");
  return success();
}

static LogicalResult
verifyQ15ResamplingProfile(Operation *op, RankedTensorType inputType, RankedTensorType initType,
                           ondrix::ondsp::FixedAttr numeric, ondrix::ondsp::ProductAttr product,
                           ondrix::ondsp::AccType accumulator, ondrix::ondsp::FixedAttr destination,
                           ondrix::ondsp::RoundingMode rounding) {
  if (failed(verifyEstablishedRounding(op, rounding, "Q15 resampling")))
    return failure();
  auto accumulatorStorage = dyn_cast<IntegerType>(accumulator.getStorage());
  if (!ondrix::ondsp::isSignedQ15(numeric) || !ondrix::ondsp::isFullProduct(product) ||
      !accumulatorStorage || accumulatorStorage.getWidth() < 32 ||
      accumulator.getSignedness() != ondrix::ondsp::Signedness::Signed ||
      accumulator.getFrac() != 30)
    return op->emitOpError(
        "supports only signed Q15/full with a signed frac30 accumulator of at least 32 bits");
  if (destination != numeric)
    return op->emitOpError("destination policy must match the signed Q15 input format");
  if (inputType.getElementType() != numeric.getStorage())
    return op->emitOpError("input and coefficient element type must match Q15 storage");
  if (initType.getElementType() != destination.getStorage())
    return op->emitOpError("init and result element type must match destination storage");
  return verifyFixedReductionResult(op, accumulator, numeric, product);
}

// The export attributes are present exactly on the fixed path and absent
// exactly on the floating-point path; both directions are checked because
// either mismatch would leave a lowering dereferencing an absent policy. The
// product attribute is covered by verifyProductPolicy at the operation entry.
template <typename OpTy>
static LogicalResult verifyResamplingNumericProfile(OpTy op, RankedTensorType inputType,
                                                    RankedTensorType initType) {
  auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  if (!fixed) {
    if (op.getAccumulator() || op.getDst() || op.getRounding() || op.getOverflow())
      return op.emitOpError(
          "floating-point resampling must not specify a fixed-point export policy");
    auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "resampling")))
      return failure();
    if (inputType.getElementType() != fp.getFormat() || initType.getElementType() != fp.getFormat())
      return op.emitOpError(
          "floating-point input, coefficients, init, and result must match format");
    return success();
  }
  if (!op.getAccumulator() || !op.getDst() || !op.getRounding() || !op.getOverflow())
    return op.emitOpError(
        "fixed resampling requires accumulator, dst, rounding, and overflow attributes");
  return verifyQ15ResamplingProfile(op, inputType, initType, fixed, *op.getProduct(),
                                    *op.getAccumulator(), *op.getDst(), *op.getRounding());
}

static LogicalResult verifyFirDecimateDomain(FirDecimateOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coefficientType = op.getCoeffs().getType();
  RankedTensorType initType = op.getInit().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, coefficientType, initType})))
    return failure();
  if (inputType.getRank() != 1 || coefficientType.getRank() != 1 || initType.getRank() != 1)
    return op.emitOpError("requires rank-1 input, coefficient, and init tensors");
  if (inputType.getElementType() != coefficientType.getElementType())
    return op.emitOpError("input and coefficient element types must match");
  int64_t factor = op.getFactorAttr().getValue().getSExtValue();
  if (factor < 2)
    return op.emitOpError("requires factor at least 2");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coefficientLength = coefficientType.getDimSize(0);
  int64_t outputLength = initType.getDimSize(0);
  if (!ShapedType::isDynamic(coefficientLength) && coefficientLength == 0)
    return op.emitOpError("requires at least one coefficient");
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coefficientLength)) {
    if (inputLength < coefficientLength)
      return op.emitOpError("input must cover one complete coefficient window");
    int64_t expectedOutputLength = (inputLength - coefficientLength) / factor + 1;
    if (!ShapedType::isDynamic(outputLength) && outputLength != expectedOutputLength)
      return op.emitOpError() << "result length must be " << expectedOutputLength;
  }

  return verifyResamplingNumericProfile(op, inputType, initType);
}

static LogicalResult verifyFirInterpolateDomain(FirInterpolateOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coefficientType = op.getCoeffs().getType();
  RankedTensorType initType = op.getInit().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, coefficientType, initType})))
    return failure();
  if (inputType.getRank() != 1 || coefficientType.getRank() != 1 || initType.getRank() != 1)
    return op.emitOpError("requires rank-1 input, coefficient, and init tensors");
  if (inputType.getElementType() != coefficientType.getElementType())
    return op.emitOpError("input and coefficient element types must match");
  int64_t factor = op.getFactorAttr().getValue().getSExtValue();
  if (factor != 2)
    return op.emitOpError("first executable profile requires factor 2");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coefficientLength = coefficientType.getDimSize(0);
  int64_t outputLength = initType.getDimSize(0);
  if (!ShapedType::isDynamic(inputLength) && inputLength == 0)
    return op.emitOpError("requires at least one input sample");
  if (!ShapedType::isDynamic(coefficientLength) && coefficientLength == 0)
    return op.emitOpError("requires at least one coefficient");
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coefficientLength)) {
    int64_t inputIntervals = inputLength - 1;
    int64_t maximumExtent = std::numeric_limits<int64_t>::max();
    if (inputIntervals > (maximumExtent - coefficientLength) / factor)
      return op.emitOpError("result length exceeds the indexable extent range");
    int64_t expectedOutputLength = inputIntervals * factor + coefficientLength;
    if (!ShapedType::isDynamic(outputLength) && outputLength != expectedOutputLength)
      return op.emitOpError() << "result length must be " << expectedOutputLength;
  }

  return verifyResamplingNumericProfile(op, inputType, initType);
}

static LogicalResult verifyConv1DDomain(Conv1DOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType kernelType = op.getKernel().getType();
  RankedTensorType initType = op.getInit().getType();
  if (failed(verifyUnencodedTensorTypes(op, {inputType, kernelType, initType})))
    return failure();
  if (inputType.getRank() != 1 || kernelType.getRank() != 1 || initType.getRank() != 1)
    return op.emitOpError("requires rank-1 input, kernel, and init tensors");
  if (inputType.getElementType() != kernelType.getElementType())
    return op.emitOpError("input and kernel element types must match");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t kernelLength = kernelType.getDimSize(0);
  int64_t outputLength = initType.getDimSize(0);
  if (!ShapedType::isDynamic(kernelLength) && kernelLength == 0)
    return op.emitOpError("requires at least one kernel element");
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(kernelLength)) {
    if (inputLength < kernelLength)
      return op.emitOpError("input must cover one complete kernel window");
    int64_t expectedOutputLength = inputLength - kernelLength + 1;
    if (!ShapedType::isDynamic(outputLength) && outputLength != expectedOutputLength)
      return op.emitOpError() << "result length must be " << expectedOutputLength;
  }

  Type inputElement = inputType.getElementType();
  Type outputElement = initType.getElementType();
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (inputElement != fixed.getStorage())
      return op.emitOpError("input and kernel element type must match fixed numeric storage");
    if (!op.getAccumulator() || !op.getDst() || !op.getRounding() || !op.getOverflow())
      return op.emitOpError(
          "fixed conv1d requires accumulator, dst, rounding, and overflow attributes");
    if (failed(verifyEstablishedRounding(op, *op.getRounding(), "fixed conv1d")))
      return failure();
    if (failed(verifyFixedReductionResult(op, *op.getAccumulator(), fixed, *op.getProduct())))
      return failure();
    if (op.getAccumulator()->getSignedness() != op.getDst()->getSignedness())
      return op.emitOpError("accumulator and destination signedness must match");
    if (op.getAccumulator()->getFrac() < op.getDst()->getFrac())
      return op.emitOpError("destination frac must not exceed accumulator frac");
    if (outputElement != op.getDst()->getStorage())
      return op.emitOpError("init and result element type must match destination storage");
    return success();
  }

  if (op.getAccumulator() || op.getDst() || op.getRounding() || op.getOverflow())
    return op.emitOpError(
        "floating-point conv1d must not specify fixed-point accumulator or export policy");
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "conv1d")))
    return failure();
  if (inputElement != fp.getFormat() || outputElement != fp.getFormat())
    return op.emitOpError("floating-point input, kernel, init, and result must match format");
  return success();
}

static LogicalResult verifyFirStreamDomain(FirStreamOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coefficientType = op.getCoeffs().getType();
  RankedTensorType stateType = op.getState().getType();
  RankedTensorType outputType = op.getOutput().getType();
  RankedTensorType nextStateType = op.getNextState().getType();
  if (failed(verifyUnencodedTensorTypes(
          op, {inputType, coefficientType, stateType, outputType, nextStateType})))
    return failure();
  if (inputType.getRank() != 1 || coefficientType.getRank() != 1 || stateType.getRank() != 1 ||
      outputType.getRank() != 1 || nextStateType.getRank() != 1)
    return op.emitOpError("requires rank-1 input, coefficients, state, and results");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coefficientLength = coefficientType.getDimSize(0);
  int64_t stateLength = stateType.getDimSize(0);
  int64_t outputLength = outputType.getDimSize(0);
  int64_t nextStateLength = nextStateType.getDimSize(0);
  if (!ShapedType::isDynamic(coefficientLength) && coefficientLength == 0)
    return op.emitOpError("requires at least one coefficient");
  if (!ShapedType::isDynamic(coefficientLength)) {
    int64_t expectedStateLength = coefficientLength - 1;
    if (!ShapedType::isDynamic(stateLength) && stateLength != expectedStateLength)
      return op.emitOpError() << "state length must be " << expectedStateLength;
    if (!ShapedType::isDynamic(nextStateLength) && nextStateLength != expectedStateLength)
      return op.emitOpError() << "next-state length must be " << expectedStateLength;
  }
  if (!ShapedType::isDynamic(stateLength) && !ShapedType::isDynamic(nextStateLength) &&
      stateLength != nextStateLength)
    return op.emitOpError("next-state length must equal state length");
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(outputLength) &&
      inputLength != outputLength)
    return op.emitOpError("output length must equal input chunk length");

  Type inputElement = inputType.getElementType();
  if (coefficientType.getElementType() != inputElement ||
      stateType.getElementType() != inputElement || nextStateType.getElementType() != inputElement)
    return op.emitOpError(
        "input, coefficients, state, and next state must have matching element types");

  Type outputElement = outputType.getElementType();
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (inputElement != fixed.getStorage())
      return op.emitOpError("stream sample element type must match fixed numeric storage");
    if (!op.getAccumulator() || !op.getDst() || !op.getRounding() || !op.getOverflow())
      return op.emitOpError(
          "fixed FIR stream requires accumulator, dst, rounding, and overflow attributes");
    if (failed(verifyEstablishedRounding(op, *op.getRounding(), "fixed FIR stream")))
      return failure();
    if (failed(verifyFixedReductionResult(op, *op.getAccumulator(), fixed, *op.getProduct())))
      return failure();
    if (op.getAccumulator()->getSignedness() != op.getDst()->getSignedness())
      return op.emitOpError("accumulator and destination signedness must match");
    if (op.getAccumulator()->getFrac() < op.getDst()->getFrac())
      return op.emitOpError("destination frac must not exceed accumulator frac");
    if (outputElement != op.getDst()->getStorage())
      return op.emitOpError("output element type must match destination storage");
    return success();
  }

  if (op.getAccumulator() || op.getDst() || op.getRounding() || op.getOverflow())
    return op.emitOpError(
        "floating-point FIR stream must not specify fixed-point accumulator or export policy");
  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "FIR stream")))
    return failure();
  if (inputElement != fp.getFormat() || outputElement != fp.getFormat())
    return op.emitOpError("floating-point stream values must match numeric format");
  return success();
}

static LogicalResult verifySosTensorLayout(Operation *op, RankedTensorType inputType,
                                           RankedTensorType coefficientType,
                                           RankedTensorType scaleType, RankedTensorType stateType) {
  if (failed(verifyUnencodedTensorTypes(op, {inputType, coefficientType, scaleType, stateType})))
    return failure();
  if (inputType.getRank() != 1 || coefficientType.getRank() != 2 || scaleType.getRank() != 1 ||
      stateType.getRank() != 2)
    return op->emitOpError("requires rank-1 input/scales and rank-2 coefficients/state tensors");
  if (coefficientType.getDimSize(1) != 5)
    return op->emitOpError("coefficient trailing dimension must be statically 5");
  if (stateType.getDimSize(1) != 2)
    return op->emitOpError("state trailing dimension must be statically 2");

  int64_t coefficientSections = coefficientType.getDimSize(0);
  int64_t scaleSections = scaleType.getDimSize(0);
  int64_t stateSections = stateType.getDimSize(0);
  if (!ShapedType::isDynamic(coefficientSections) && coefficientSections == 0)
    return op->emitOpError("requires at least one second-order section");
  auto staticallyDisagree = [](int64_t lhs, int64_t rhs) {
    return !ShapedType::isDynamic(lhs) && !ShapedType::isDynamic(rhs) && lhs != rhs;
  };
  if (staticallyDisagree(coefficientSections, scaleSections) ||
      staticallyDisagree(coefficientSections, stateSections) ||
      staticallyDisagree(scaleSections, stateSections))
    return op->emitOpError("coefficient, scale, and state section counts must match");
  return success();
}

static LogicalResult verifySosFilterTdf2Domain(SosFilterTdf2Op op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coefficientType = op.getCoeffs().getType();
  RankedTensorType scaleType = op.getScales().getType();
  RankedTensorType stateType = op.getState().getType();
  if (failed(verifySosTensorLayout(op, inputType, coefficientType, scaleType, stateType)))
    return failure();

  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (!fp || !fp.getFormat().isF32())
    return op.emitOpError("currently requires an f32 numeric policy");
  // Admission follows the legal set, not the transform list (rule:
  // docs/f32-contract-evidence.md). A biquad body has fusable multiply-adds,
  // so its set exceeds one graph and needs a realization gate for whatever is
  // selected. That is an evidence gap and not a property of recursions - lms
  // is a recursion too and is admitted.
  if (fp.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return op.emitOpError("fast has no realization gate here; use off or fma");
  if (inputType.getElementType() != fp.getFormat() ||
      coefficientType.getElementType() != fp.getFormat() ||
      scaleType.getElementType() != fp.getFormat() || stateType.getElementType() != fp.getFormat())
    return op.emitOpError("input, coefficients, scales, state, and results must use f32");
  return success();
}

static LogicalResult verifySosFilterDf2FixedDomain(SosFilterDf2FixedOp op) {
  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType coefficientType = op.getCoeffs().getType();
  RankedTensorType scaleType = op.getScales().getType();
  RankedTensorType stateType = op.getState().getType();
  if (failed(verifySosTensorLayout(op, inputType, coefficientType, scaleType, stateType)))
    return failure();

  ondrix::ondsp::FixedAttr numeric = op.getNumeric();
  Type storage = numeric.getStorage();
  if (inputType.getElementType() != storage || coefficientType.getElementType() != storage ||
      scaleType.getElementType() != storage || stateType.getElementType() != storage)
    return op.emitOpError(
        "input, coefficients, scales, state, and results must match numeric storage");
  if (!ondrix::ondsp::isFullProduct(op.getProduct()))
    return op.emitOpError("supports only exact full products");

  ondrix::ondsp::AccType accumulator = op.getAccumulator();
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return op.emitOpError("algorithm accumulator must be single-lane; lane batching is a lowering "
                          "decision, not part of the algorithm contract");
  bool isQ15 = ondrix::ondsp::isSignedQ15(numeric) &&
               ondrix::ondsp::isSignedI40Frac30Accumulator(accumulator);
  bool isQ31 = ondrix::ondsp::isSignedQ31(numeric) &&
               ondrix::ondsp::isSignedI64Frac62Accumulator(accumulator);
  if (!isQ15 && !isQ31)
    return op.emitOpError(
        "supports only signed Q15/full with i40/frac30 accumulator or signed Q31/full with "
        "i64/frac62 accumulator");
  if (failed(verifyEstablishedRounding(op, op.getStateRounding(), "state_rounding")) ||
      failed(verifyEstablishedRounding(op, op.getOutputRounding(), "output_rounding")))
    return failure();
  return success();
}

static OpFoldResult getTensorDim(OpBuilder &builder, Location loc, Value tensor, int64_t dim) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));
  return builder.create<tensor::DimOp>(loc, tensor, dim).getResult();
}

static OpFoldResult addFirInputHalo(OpBuilder &builder, Location loc, OpFoldResult outputTileSize,
                                    OpFoldResult coefficientLength) {
  auto getConstant = [](OpFoldResult value) -> std::optional<int64_t> {
    if (auto attribute = value.dyn_cast<Attribute>())
      return cast<IntegerAttr>(attribute).getInt();
    return std::nullopt;
  };
  std::optional<int64_t> staticTileSize = getConstant(outputTileSize);
  std::optional<int64_t> staticCoefficientLength = getConstant(coefficientLength);
  if (staticTileSize && staticCoefficientLength)
    return builder.getIndexAttr(*staticTileSize + (*staticCoefficientLength - 1));

  auto materialize = [&](OpFoldResult value) -> Value {
    if (auto dynamic = value.dyn_cast<Value>())
      return dynamic;
    return builder.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(value.get<Attribute>()).getInt());
  };
  Value tileSize = materialize(outputTileSize);
  Value coefficients = materialize(coefficientLength);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value halo = builder.create<arith::SubIOp>(loc, coefficients, one);
  return builder.create<arith::AddIOp>(loc, tileSize, halo).getResult();
}

static LogicalResult verifyDotDomain(DotOp op) {
  if (isa<RankedTensorType>(op.getLhs().getType()) || isa<RankedTensorType>(op.getRhs().getType()))
    return op.emitOpError(
        "tensor dot operands have no executable consumer; use memrefs or fixed vectors");

  auto lhsShaped = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhsShaped = dyn_cast<ShapedType>(op.getRhs().getType());
  if (static_cast<bool>(lhsShaped) != static_cast<bool>(rhsShaped))
    return op.emitOpError("requires either two scalar operands or two rank-1 shaped operands");

  Type lhsElement = op.getLhs().getType();
  Type rhsElement = op.getRhs().getType();
  if (lhsShaped) {
    if (!lhsShaped.hasRank() || !rhsShaped.hasRank() || lhsShaped.getRank() != 1 ||
        rhsShaped.getRank() != 1)
      return op.emitOpError("shaped operands must be rank-1");
    if (ondrix::isScalableVectorType(op.getLhs().getType()) ||
        ondrix::isScalableVectorType(op.getRhs().getType()))
      return op.emitOpError("scalable vector operands are not supported");
    if (isa<ondrix::ondsp::FpAttr>(op.getNumeric()) &&
        (isa<VectorType>(op.getLhs().getType()) || isa<VectorType>(op.getRhs().getType())))
      return op.emitOpError("floating-point vector dot operands have no executable consumer");
    lhsElement = lhsShaped.getElementType();
    rhsElement = rhsShaped.getElementType();

    int64_t lhsLength = lhsShaped.getDimSize(0);
    int64_t rhsLength = rhsShaped.getDimSize(0);
    if (!ShapedType::isDynamic(lhsLength) && !ShapedType::isDynamic(rhsLength) &&
        lhsLength != rhsLength)
      return op.emitOpError("shaped operands must have equal static lengths");
  }

  if (lhsElement != rhsElement)
    return op.emitOpError("operand element types must match");

  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (lhsElement != fixed.getStorage())
      return op.emitOpError("operand element type must match fixed numeric storage type");
    return verifyFixedReductionResult(op, op.getResult().getType(), fixed, *op.getProduct());
  }

  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (failed(ondrix::ondsp::verifyExecutableFpFormat(op, fp, "dot")))
    return failure();
  if (lhsElement != fp.getFormat() || op.getResult().getType() != fp.getFormat())
    return op.emitOpError("floating-point dot operands and result must match numeric format");
  return success();
}

} // namespace

void FirOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getInput(), effects);
  addMemRefReadEffect(getCoeffs(), effects);
}

Speculation::Speculatability FirOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

void DotOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getLhs(), effects);
  addMemRefReadEffect(getRhs(), effects);
}

Speculation::Speculatability DotOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getLhs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getRhs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirWindow(*this);
}

Speculation::Speculatability FirFilterOp::getSpeculatability() {
  return (getOutputOrigin() || ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getInit().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirFilterOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirFilterDomain(*this);
}

Speculation::Speculatability FirDecimateOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getInit().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirDecimateOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirDecimateDomain(*this);
}

// One shape and format rule for the whole elementwise family: matching static
// rank-1 Q15 tensors. Passing every operand keeps a binary member from
// checking only its left side.
static LogicalResult verifyElementwiseQ15Domain(Operation *op, Attribute numeric,
                                                llvm::ArrayRef<RankedTensorType> types) {
  if (failed(verifySignedFixedFormat(op, numeric, 16, 15, "numeric")))
    return failure();
  if (failed(verifyUnencodedTensorTypes(op, types)))
    return failure();
  int64_t extent =
      types.front().getRank() == 1 ? types.front().getDimSize(0) : ShapedType::kDynamic;
  bool ok = extent != ShapedType::kDynamic && extent >= 1 && extent <= 4096;
  for (RankedTensorType type : types)
    ok &= type.getRank() == 1 && type.getDimSize(0) == extent &&
          type.getElementType().isSignlessInteger(16);
  if (!ok)
    return op->emitOpError("executable elementwise operations require matching static "
                           "tensor<Nxi16> operands and result with N in [1, 4096]");
  return success();
}

// The four declared tie rules are all admissible at an elementwise
// requantization: round_shift implements each of them and the object gate
// runs all four.
static LogicalResult verifyElementwiseRounding(Operation *op,
                                               ondrix::ondsp::RoundingMode rounding) {
  switch (rounding) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
  case ondrix::ondsp::RoundingMode::TowardZero:
  case ondrix::ondsp::RoundingMode::NearestEven:
  case ondrix::ondsp::RoundingMode::NearestTiesPositive:
    return success();
  }
  return op->emitOpError("unsupported rounding mode");
}

LogicalResult AddOp::verify() {
  return verifyElementwiseQ15Domain(
      getOperation(), getNumeric(),
      {getLhs().getType(), getRhs().getType(), getResult().getType()});
}

LogicalResult SubOp::verify() {
  return verifyElementwiseQ15Domain(
      getOperation(), getNumeric(),
      {getLhs().getType(), getRhs().getType(), getResult().getType()});
}

LogicalResult MultOp::verify() {
  if (failed(verifyElementwiseRounding(getOperation(), getRounding())))
    return failure();
  return verifyElementwiseQ15Domain(
      getOperation(), getNumeric(),
      {getLhs().getType(), getRhs().getType(), getResult().getType()});
}

LogicalResult AbsOp::verify() {
  return verifyElementwiseQ15Domain(getOperation(), getNumeric(),
                                    {getInput().getType(), getResult().getType()});
}

LogicalResult NegateOp::verify() {
  return verifyElementwiseQ15Domain(getOperation(), getNumeric(),
                                    {getInput().getType(), getResult().getType()});
}

LogicalResult OffsetOp::verify() {
  int64_t bias = getBiasAttr().getInt();
  if (bias < -32768 || bias > 32767)
    return emitOpError("offset bias must be a raw signed Q1.15 value in [-32768, 32767]");
  return verifyElementwiseQ15Domain(getOperation(), getNumeric(),
                                    {getInput().getType(), getResult().getType()});
}

LogicalResult ShiftOp::verify() {
  int64_t amount = getAmountAttr().getInt();
  if (amount < -15 || amount > 15)
    return emitOpError("shift amount must lie in [-15, 15]");
  if (failed(verifyElementwiseRounding(getOperation(), getRounding())))
    return failure();
  return verifyElementwiseQ15Domain(getOperation(), getNumeric(),
                                    {getInput().getType(), getResult().getType()});
}

int64_t CicDecimateOp::getGrowthBits() {
  return getStages() * llvm::Log2_64(uint64_t(getRate()) * uint64_t(getDelay()));
}

LogicalResult CicDecimateOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  if (getRounding() != ondrix::ondsp::RoundingMode::NearestEven &&
      getRounding() != ondrix::ondsp::RoundingMode::NearestTiesPositive)
    return emitOpError("cic decimation requires nearest_even or nearest_ties_positive rounding");
  int64_t stages = getStages();
  int64_t rate = getRate();
  int64_t delay = getDelay();
  if (stages < 1 || stages > 8)
    return emitOpError("cic decimation requires stages in [1, 8]");
  if (rate < 2 || rate > 4096 || !llvm::isPowerOf2_64(uint64_t(rate)))
    return emitOpError("cic decimation requires a power-of-two rate in [2, 4096]");
  if (delay != 1 && delay != 2)
    return emitOpError("cic decimation requires a differential delay of 1 or 2");
  // W = 16 + G must stay inside the widest carrier the fixed lowerings use.
  if (16 + getGrowthBits() > 64)
    return emitOpError("cic decimation requires stages * log2(rate * delay) <= 48");
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent == ShapedType::kDynamic || resultExtent == ShapedType::kDynamic ||
      resultExtent < 1 || resultExtent > 4096 || inputExtent != resultExtent * rate ||
      !inputType.getElementType().isSignlessInteger(16) ||
      !resultType.getElementType().isSignlessInteger(16))
    return emitOpError("executable cic decimation requires static tensor<(R*L)xi16> input and "
                       "tensor<Lxi16> result with L in [1, 4096]");
  return success();
}

Speculation::Speculatability FirInterpolateOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getInit().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirInterpolateOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirInterpolateDomain(*this);
}

Speculation::Speculatability Conv1DOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getKernel().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getInit().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult Conv1DOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyConv1DDomain(*this);
}

Speculation::Speculatability FirStreamOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getState().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getOutput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getNextState().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirStreamOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirStreamDomain(*this);
}

Speculation::Speculatability SosFilterTdf2Op::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getScales().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getState().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult SosFilterTdf2Op::verify() { return verifySosFilterTdf2Domain(*this); }

Speculation::Speculatability SosFilterDf2FixedOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getScales().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getState().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult SosFilterDf2FixedOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifySosFilterDf2FixedDomain(*this);
}

SmallVector<utils::IteratorType> FirFilterOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel};
}

SmallVector<Range> FirFilterOp::getIterationDomain(OpBuilder &builder) {
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  return {{zero, getTensorDim(builder, getLoc(), getInit(), 0), one}};
}

FailureOr<TilingResult> FirFilterOp::getTiledImplementation(OpBuilder &builder,
                                                            ArrayRef<OpFoldResult> offsets,
                                                            ArrayRef<OpFoldResult> sizes) {
  if (offsets.size() != 1 || sizes.size() != 1)
    return failure();

  Location loc = getLoc();
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<Value> tiledOperands;
  Value tiledInit;
  if (getBoundary() == FirBoundaryMode::Valid) {
    OpFoldResult coefficientLength = getTensorDim(builder, loc, getCoeffs(), 0);
    OpFoldResult inputTileSize = addFirInputHalo(builder, loc, sizes.front(), coefficientLength);
    Value tiledInput = builder.create<tensor::ExtractSliceOp>(loc, getInput(), offsets,
                                                              ArrayRef<OpFoldResult>{inputTileSize},
                                                              ArrayRef<OpFoldResult>{one});
    tiledInit = builder.create<tensor::ExtractSliceOp>(loc, getInit(), offsets, sizes,
                                                       ArrayRef<OpFoldResult>{one});
    tiledOperands = {tiledInput, getCoeffs(), tiledInit};
  } else {
    tiledInit = builder.create<tensor::ExtractSliceOp>(loc, getInit(), offsets, sizes,
                                                       ArrayRef<OpFoldResult>{one});
    Value outputOrigin = getValueOrCreateConstantIndexOp(builder, loc, offsets.front());
    if (getOutputOrigin())
      outputOrigin = builder.create<arith::AddIOp>(loc, getOutputOrigin(), outputOrigin);
    tiledOperands = {getInput(), getCoeffs(), tiledInit, outputOrigin};
  }
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), TypeRange{tiledInit.getType()}, tiledOperands);
  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult FirFilterOp::getResultTilePosition(OpBuilder &builder, unsigned resultNumber,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes,
                                                 SmallVector<OpFoldResult> &resultOffsets,
                                                 SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0 || offsets.size() != 1 || sizes.size() != 1)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

FailureOr<TilingResult> FirFilterOp::generateResultTileValue(OpBuilder &builder,
                                                             unsigned resultNumber,
                                                             ArrayRef<OpFoldResult> offsets,
                                                             ArrayRef<OpFoldResult> sizes) {
  if (resultNumber != 0)
    return failure();
  return getTiledImplementation(builder, offsets, sizes);
}

LogicalResult DotOp::verify() {
  if (failed(ondrix::ondsp::verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyDotDomain(*this);
}

// The value domain runs before the numeric policy in every FFT-family
// verifier below. The policy is layout-driven now, so a Q15-only operation
// must reject an unsupported layout with its own diagnostic before the shared
// policy starts reporting the width rules of a profile that operation does
// not implement.
LogicalResult ButterflyOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifyButterflyValueDomain(*this)))
    return failure();
  return ondrix::ondsp::verifyPackedButterflyPolicy(*this, getLayout(), getNumeric(), getProduct(),
                                                    getProductScale(), getOutputScale());
}

LogicalResult CfftOp::verify() {
  if (failed(verifyCfftValueDomain(*this)))
    return failure();
  return ondrix::ondsp::verifyPackedButterflyPolicy(*this, getLayout(), getNumeric(), getProduct(),
                                                    getProductScale(), getOutputScale());
}

LogicalResult RfftOp::verify() {
  if (failed(verifyRfftValueDomain(*this)))
    return failure();
  return ondrix::ondsp::verifyPackedButterflyPolicy(*this, getLayout(), getNumeric(), getProduct(),
                                                    getProductScale(), getOutputScale());
}

LogicalResult IrfftOp::verify() {
  if (failed(verifyIrfftValueDomain(*this)))
    return failure();
  return ondrix::ondsp::verifyPackedButterflyPolicy(*this, getLayout(), getNumeric(), getProduct(),
                                                    getProductScale(), getOutputScale());
}

LogicalResult RfftRadix4SplitOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getInputNumeric(), 16, 15, "input_numeric")))
    return failure();
  if (failed(verifySignedFixedFormat(getOperation(), getOutputNumeric(), 16, 10, "output_numeric")))
    return failure();
  if (!ondrix::ondsp::isFullProduct(getProduct()))
    return emitOpError("executable radix-4 split RFFT requires product = #ondsp.product<full>");
  return verifyRfftRadix4SplitValueDomain(*this);
}

LogicalResult DctOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getInputNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "DCT")))
      return failure();
    // An f32 output needs no rescaled reading: the two numeric attributes
    // name the same format.
    if (getOutputNumeric() != getInputNumeric())
      return emitOpError("floating-point DCT output_numeric must equal input_numeric");
  } else if (failed(verifySignedFixedFormat(getOperation(), getInputNumeric(), 16, 15,
                                            "input_numeric"))) {
    return failure();
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t extent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  if (extent < 4 || extent > 64 || !llvm::isPowerOf2_64(extent) || inputType != resultType ||
      inputType.getElementType() != element) {
    llvm::StringRef name = fp ? "f32" : "i16";
    return emitOpError() << "executable DCT requires matching tensor<Nx" << name
                         << "> input and result with power-of-two N in [4, 64]";
  }
  if (fp)
    return success();
  unsigned stageCount = llvm::Log2_64(extent);
  if (failed(verifySignedFixedFormat(getOperation(), getOutputNumeric(), 16, 14 - stageCount,
                                     "output_numeric")))
    return failure();
  return success();
}

LogicalResult MovingAverageOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "moving average")))
      return failure();
  } else if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric"))) {
    return failure();
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t window = getWindow();
  if (window < 2 || window > 64)
    return emitOpError("executable moving average requires a window in [2, 64]");
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  if (inputExtent == ShapedType::kDynamic || inputExtent < window ||
      resultExtent != inputExtent - window + 1 || inputType.getElementType() != element ||
      resultType.getElementType() != element) {
    llvm::StringRef name = fp ? "f32" : "i16";
    return emitOpError() << "executable moving average requires static tensor<Nx" << name
                         << "> input and tensor<(N-K+1)x" << name << "> result with N >= K";
  }
  return success();
}

static LogicalResult verifyTrigValueDomain(Operation *op, Attribute numeric,
                                           ondrix::ondsp::RoundingMode rounding,
                                           RankedTensorType inputType,
                                           RankedTensorType resultType) {
  if (failed(verifySignedFixedFormat(op, numeric, 16, 15, "numeric")))
    return failure();
  if (rounding != ondrix::ondsp::RoundingMode::NearestEven)
    return op->emitOpError("trigonometric operations require nearest_even rounding");
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent == ShapedType::kDynamic || inputExtent < 1 || inputExtent > 4096 ||
      resultExtent != inputExtent || !inputType.getElementType().isSignlessInteger(16) ||
      !resultType.getElementType().isSignlessInteger(16))
    return op->emitOpError("executable trigonometric operations require matching static "
                           "tensor<Nxi16> input and result with N in [1, 4096]");
  return success();
}

// The two readings are opposite ends of the same pair, so one helper states
// which is which and neither operation can quietly declare the other's.
static LogicalResult verifyExponentialDomain(Operation *op, ondrix::ondsp::FixedAttr numeric,
                                             ondrix::ondsp::FixedAttr outputNumeric,
                                             ondrix::ondsp::RoundingMode rounding,
                                             RankedTensorType inputType,
                                             RankedTensorType resultType, bool inputIsMagnitude) {
  auto isMagnitude = [](ondrix::ondsp::FixedAttr attr) {
    return attr.getSignedness() == ondrix::ondsp::Signedness::Unsigned &&
           attr.getStorage().isSignlessInteger(16) && attr.getFrac() == 16;
  };
  auto isExponent = [](ondrix::ondsp::FixedAttr attr) {
    return attr.getSignedness() == ondrix::ondsp::Signedness::Signed &&
           attr.getStorage().isSignlessInteger(16) && attr.getFrac() == 11;
  };
  bool ok = inputIsMagnitude ? (isMagnitude(numeric) && isExponent(outputNumeric))
                             : (isExponent(numeric) && isMagnitude(outputNumeric));
  if (!ok)
    return op->emitOpError() << "requires "
                             << (inputIsMagnitude
                                     ? "an unsigned Q0.16 input and a signed Q5.11 result"
                                     : "a signed Q5.11 input and an unsigned Q0.16 result")
                             << ", each declared in its own numeric attribute";
  if (rounding != ondrix::ondsp::RoundingMode::NearestEven)
    return op->emitOpError("exponential operations require nearest_even rounding");
  if (failed(verifyUnencodedTensorTypes(op, {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent == ShapedType::kDynamic || inputExtent < 1 || inputExtent > 4096 ||
      resultExtent != inputExtent || !inputType.getElementType().isSignlessInteger(16) ||
      !resultType.getElementType().isSignlessInteger(16))
    return op->emitOpError("executable exponential operations require matching static "
                           "tensor<Nxi16> input and result with N in [1, 4096]");
  return success();
}

LogicalResult Log2Op::verify() {
  return verifyExponentialDomain(getOperation(), getNumeric(), getOutputNumeric(), getRounding(),
                                 getInput().getType(), getResult().getType(),
                                 /*inputIsMagnitude=*/true);
}

LogicalResult Exp2Op::verify() {
  return verifyExponentialDomain(getOperation(), getNumeric(), getOutputNumeric(), getRounding(),
                                 getInput().getType(), getResult().getType(),
                                 /*inputIsMagnitude=*/false);
}

LogicalResult SineOp::verify() {
  return verifyTrigValueDomain(getOperation(), getNumeric(), getRounding(), getInput().getType(),
                               getResult().getType());
}

LogicalResult CosineOp::verify() {
  return verifyTrigValueDomain(getOperation(), getNumeric(), getRounding(), getInput().getType(),
                               getResult().getType());
}

LogicalResult GoertzelOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "goertzel")))
      return failure();
    if (getRounding())
      return emitOpError("floating-point goertzel rounds at no declared boundary of its own");
  } else {
    if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
      return failure();
    if (!getRounding() || *getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return emitOpError("goertzel requires nearest_even rounding");
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType energyType = getEnergy().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, energyType})))
    return failure();
  int64_t extent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t energyExtent =
      energyType.getRank() == 1 ? energyType.getDimSize(0) : ShapedType::kDynamic;
  // The fixed energy is the exact integer the recursion produces and needs a
  // wider storage than its input; the f32 energy is the same format.
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  Type energyElement = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 64));
  if (extent == ShapedType::kDynamic || extent < 2 || extent > 4096 ||
      inputType.getElementType() != element || energyExtent != 1 ||
      energyType.getElementType() != energyElement) {
    llvm::StringRef name = fp ? "f32" : "i16";
    llvm::StringRef energyName = fp ? "f32" : "i64";
    return emitOpError() << "executable goertzel requires static tensor<Nx" << name
                         << "> input with N in [2, 4096] and tensor<1x" << energyName << "> energy";
  }
  int64_t bin = getBin();
  if (bin < 0 || bin > extent / 2)
    return emitOpError("goertzel bin must lie in [0, N/2]");
  return success();
}

LogicalResult MatmulOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "matmul")))
      return failure();
    if (getRounding())
      return emitOpError("floating-point matmul has no requantization boundary to round");
  } else {
    if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
      return failure();
    if (!getRounding() || *getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return emitOpError("matmul requires nearest_even rounding");
  }
  RankedTensorType lhsType = getLhs().getType();
  RankedTensorType rhsType = getRhs().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {lhsType, rhsType, resultType})))
    return failure();
  Type element =
      fp ? Type(fp.getFormat()) : Type(cast<ondrix::ondsp::FixedAttr>(getNumeric()).getStorage());
  auto isStaticMatrix = [&](RankedTensorType type) {
    return type.getRank() == 2 && type.hasStaticShape() && type.getElementType() == element;
  };
  auto inRange = [](int64_t dim) { return dim >= 1 && dim <= 64; };
  if (!isStaticMatrix(lhsType) || !isStaticMatrix(rhsType) || !isStaticMatrix(resultType) ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0) ||
      resultType.getDimSize(0) != lhsType.getDimSize(0) ||
      resultType.getDimSize(1) != rhsType.getDimSize(1) || !inRange(lhsType.getDimSize(0)) ||
      !inRange(lhsType.getDimSize(1)) || !inRange(rhsType.getDimSize(1))) {
    llvm::StringRef name = fp ? "f32" : "i16";
    return emitOpError() << "executable matmul requires static tensor<MxKx" << name
                         << "> x tensor<KxNx" << name << "> -> tensor<MxNx" << name
                         << "> with M, K, N in [1, 64]";
  }
  return success();
}

LogicalResult RmsOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "rms")))
      return failure();
    if (getRounding())
      return emitOpError("floating-point rms rounds at no declared boundary of its own");
  } else {
    if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
      return failure();
    if (!getRounding() || (*getRounding() != ondrix::ondsp::RoundingMode::TowardNegative &&
                           *getRounding() != ondrix::ondsp::RoundingMode::NearestEven))
      return emitOpError("rms supports toward_negative or nearest_even rounding");
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  // The power-of-two extent exists so the fixed-point mean is a shift; an f32
  // mean divides by a representable constant at any admitted extent.
  if (inputExtent == ShapedType::kDynamic || inputExtent < 2 || inputExtent > 4096 ||
      (!fp && !llvm::isPowerOf2_64(inputExtent)) || resultExtent != 1 ||
      inputType.getElementType() != element || resultType.getElementType() != element) {
    llvm::StringRef name = fp ? "f32" : "i16";
    return emitOpError() << "executable rms requires static tensor<Nx" << name << "> input with "
                         << (fp ? "N" : "power-of-two N") << " in [2, 4096] and tensor<1x" << name
                         << "> result";
  }
  return success();
}

LogicalResult GainOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "gain")))
      return failure();
    if (getRounding())
      return emitOpError("floating-point gain has no requantization boundary to round");
    if (getGain())
      return emitOpError("floating-point gain must not specify a raw Q1.15 constant");
    if (!getFpGain())
      return emitOpError("floating-point gain requires the fp_gain constant");
  } else {
    if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
      return failure();
    if (getFpGain())
      return emitOpError("fixed gain must not specify a floating-point constant");
    if (!getRounding() || (*getRounding() != ondrix::ondsp::RoundingMode::NearestEven &&
                           *getRounding() != ondrix::ondsp::RoundingMode::NearestTiesPositive))
      return emitOpError("gain requires nearest_even or nearest_ties_positive rounding");
    if (!getGain() || getGainAttr().getInt() < -32768 || getGainAttr().getInt() > 32767)
      return emitOpError("gain constant must be a raw signed Q1.15 value in [-32768, 32767]");
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  if (inputExtent == ShapedType::kDynamic || inputExtent < 1 || inputExtent > 4096 ||
      resultExtent != inputExtent || inputType.getElementType() != element ||
      resultType.getElementType() != element) {
    llvm::StringRef name = fp ? "f32" : "i16";
    return emitOpError() << "executable gain requires matching static tensor<Nx" << name
                         << "> input and result with N in [1, 4096]";
  }
  return success();
}

LogicalResult LmsOp::verify() {
  auto fp = dyn_cast<ondrix::ondsp::FpAttr>(getNumeric());
  if (fp) {
    if (failed(ondrix::ondsp::verifyExecutableFpFormat(*this, fp, "lms")))
      return failure();
    if (getRounding())
      return emitOpError("floating-point lms rounds at no declared boundary of its own");
    if (getStepSize())
      return emitOpError("floating-point lms must not specify a raw Q1.15 step size");
    if (!getFpStepSize())
      return emitOpError("floating-point lms requires the fp_step_size constant");
    // The fixed profile admits only the non-negative raw Q1.15 range, and a
    // NaN step size fails this comparison with it.
    if (!(getFpStepSize()->convertToFloat() >= 0.0f))
      return emitOpError("floating-point lms step size must not be negative");
  } else {
    if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
      return failure();
    if (getFpStepSize())
      return emitOpError("fixed lms must not specify a floating-point step size");
    if (!getRounding() || *getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return emitOpError("lms requires nearest_even rounding");
    if (!getStepSize() || getStepSizeAttr().getInt() < 0 || getStepSizeAttr().getInt() > 32767)
      return emitOpError("lms step size must be a raw signed Q1.15 value in [0, 32767]");
  }
  RankedTensorType inputType = getInput().getType();
  RankedTensorType desiredType = getDesired().getType();
  RankedTensorType weightsType = getWeights().getType();
  RankedTensorType errorType = getError().getType();
  RankedTensorType adaptedType = getAdapted().getType();
  if (failed(verifyUnencodedTensorTypes(
          getOperation(), {inputType, desiredType, weightsType, errorType, adaptedType})))
    return failure();
  Type element = fp ? Type(fp.getFormat()) : Type(IntegerType::get(getContext(), 16));
  llvm::StringRef name = fp ? "f32" : "i16";
  auto staticExtent = [&](RankedTensorType type) {
    return type.getRank() == 1 && type.getElementType() == element ? type.getDimSize(0)
                                                                   : ShapedType::kDynamic;
  };
  int64_t samples = staticExtent(inputType);
  int64_t taps = staticExtent(weightsType);
  if (samples == ShapedType::kDynamic || samples < 1 || samples > 4096 ||
      staticExtent(desiredType) != samples || staticExtent(errorType) != samples)
    return emitOpError() << "executable lms requires matching static tensor<Nx" << name
                         << "> input, desired, and error with N in [1, 4096]";
  if (taps == ShapedType::kDynamic || taps < 1 || taps > 64 || staticExtent(adaptedType) != taps)
    return emitOpError() << "executable lms requires matching static tensor<Kx" << name
                         << "> weights and adapted weights with K in [1, 64]";
  return success();
}

LogicalResult CxMagnitudeOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  if (getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return emitOpError("executable magnitude requires packed_i16_imag_hi_real_lo layout");
  ondrix::ondsp::RoundingMode rounding = getRounding();
  if (rounding != ondrix::ondsp::RoundingMode::TowardNegative &&
      rounding != ondrix::ondsp::RoundingMode::NearestEven)
    return emitOpError("cx_magnitude supports toward_negative or nearest_even rounding");
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent == ShapedType::kDynamic || inputExtent < 1 || inputExtent > 4096 ||
      resultExtent != inputExtent || !inputType.getElementType().isSignlessInteger(32) ||
      !resultType.getElementType().isSignlessInteger(16))
    return emitOpError("executable magnitude requires tensor<Nxi32> to tensor<Nxi16> "
                       "with static N in [1, 4096]");
  return success();
}

LogicalResult CxPhaseOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  ondrix::ondsp::FixedAttr output = getOutputNumeric();
  if (output.getSignedness() != ondrix::ondsp::Signedness::Unsigned ||
      !output.getStorage().isSignlessInteger(16) || output.getFrac() != 16)
    return emitOpError("cx_phase returns the unsigned Q0.16 turn and must declare that reading");
  if (getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
    return emitOpError("executable phase requires packed_i16_imag_hi_real_lo layout");
  if (getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
    return emitOpError("cx_phase requires nearest_even rounding");
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (failed(verifyUnencodedTensorTypes(getOperation(), {inputType, resultType})))
    return failure();
  int64_t inputExtent = inputType.getRank() == 1 ? inputType.getDimSize(0) : ShapedType::kDynamic;
  int64_t resultExtent =
      resultType.getRank() == 1 ? resultType.getDimSize(0) : ShapedType::kDynamic;
  if (inputExtent == ShapedType::kDynamic || inputExtent < 1 || inputExtent > 4096 ||
      resultExtent != inputExtent || !inputType.getElementType().isSignlessInteger(32) ||
      !resultType.getElementType().isSignlessInteger(16))
    return emitOpError("executable phase requires tensor<Nxi32> to tensor<Nxi16> "
                       "with static N in [1, 4096]");
  return success();
}

LogicalResult WindowHammingOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  return verifyDesignCoefficientTensor(getOperation(), getCoefficients().getType(), 2, 4096);
}

LogicalResult WindowHannOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  return verifyDesignCoefficientTensor(getOperation(), getCoefficients().getType(), 2, 4096);
}

LogicalResult WindowBlackmanOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  return verifyDesignCoefficientTensor(getOperation(), getCoefficients().getType(), 2, 4096);
}

LogicalResult WindowKaiserOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  int64_t num = getBetaNum();
  int64_t den = getBetaDen();
  // beta in (0, 50]: positive rational, bounded so the binary64 I0 series
  // stays far from overflow (I0(50) ~ 3e20) and the documented evaluation
  // error budget holds. The sign check must run FIRST: `50 * den` on a
  // negative denominator is signed overflow (UB) long before any range
  // logic, so only proven-positive values reach the product below.
  if (num < 1 || den < 1)
    return emitOpError("kaiser beta must be a positive rational in (0, 50]");
  // When 50 * den would overflow i64, num (at most INT64_MAX) is
  // necessarily below the true product and the beta is in range — compare
  // only when the product is representable.
  bool aboveFifty = den <= std::numeric_limits<int64_t>::max() / 50 && num > 50 * den;
  if (aboveFifty)
    return emitOpError("kaiser beta must be a positive rational in (0, 50]");
  return verifyDesignCoefficientTensor(getOperation(), getCoefficients().getType(), 2, 4096);
}

LogicalResult FirDesignWindowedSincOp::verify() {
  if (failed(verifySignedFixedFormat(getOperation(), getNumeric(), 16, 15, "numeric")))
    return failure();
  if (failed(verifyDesignCoefficientTensor(getOperation(), getCoefficients().getType(), 3, 4095)))
    return failure();
  if (getCoefficients().getType().getDimSize(0) % 2 == 0)
    return emitOpError("windowed-sinc design requires an odd coefficient extent");
  int64_t num = getCutoffNum();
  int64_t den = getCutoffDen();
  if (num < 1 || den < 2 || num > (den - 1) / 2)
    return emitOpError("cutoff requires 1 <= cutoff_num and 2 * cutoff_num < cutoff_den");
  return success();
}

LogicalResult QuantizeOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyQuantizeDomain(*this);
}
