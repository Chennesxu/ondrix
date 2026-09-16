#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/GuardedFixedQuantization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

using namespace mlir;
using namespace ondrix::conversion;

namespace {

// One loop per operation: the operands' elements go through `body` and the
// result lands in a fresh tensor. The elementwise family and the width
// conversion share it.
static Value emitElementwiseLoop(Location loc, RankedTensorType resultType, ValueRange sources,
                                 llvm::function_ref<Value(ArrayRef<Value>, OpBuilder &)> body,
                                 OpBuilder &rewriter) {
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getDimSize(0));
  Value empty =
      rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{empty},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        SmallVector<Value> elements;
        for (Value source : sources)
          elements.push_back(builder.create<tensor::ExtractOp>(loc, source, position));
        Value value = body(elements, builder);
        Value inserted = builder.create<tensor::InsertOp>(loc, value, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  return loop.getResult(0);
}

// The value-preserving width change: a widening is the exact left shift by
// the width difference, a narrowing is the one requantization boundary the
// operation declares, at the same shift.
static Value emitQuantizeBody(ondrix::ir::QuantizeOp op, Value value, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto sourceFixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getSrc());
  auto destinationFixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getDst());
  if (!sourceFixed || !destinationFixed) {
    // The domain change keeps its own boundary operation.
    Type element = destinationFixed ? destinationFixed.getStorage()
                                    : cast<ondrix::ondsp::FpAttr>(op.getDst()).getFormat();
    return builder.create<ondrix::ondsp::ConvertOp>(loc, element, value, op.getSrc(), op.getDst(),
                                                    op.getRoundingAttr(), op.getOverflowAttr());
  }
  auto source = cast<IntegerType>(sourceFixed.getStorage());
  auto destination = cast<IntegerType>(destinationFixed.getStorage());
  if (destination.getWidth() > source.getWidth()) {
    Value extended = builder.create<arith::ExtSIOp>(loc, destination, value);
    Value shift = builder.create<arith::ConstantIntOp>(
        loc, destination.getWidth() - source.getWidth(), destination);
    return builder.create<arith::ShLIOp>(loc, extended, shift);
  }
  auto scale = ondrix::ondsp::ScaleAttr::get(builder.getContext(), /*preShiftLeft=*/0,
                                             source.getWidth() - destination.getWidth(),
                                             *op.getRounding(), *op.getOverflow(), destination);
  return builder.create<ondrix::ondsp::RoundShiftOp>(loc, destination, value, scale);
}

class QuantizeOpLowering final : public OpConversionPattern<ondrix::ir::QuantizeOp> {
public:
  using OpConversionPattern<ondrix::ir::QuantizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) {
      rewriter.replaceOp(op, emitQuantizeBody(op, adaptor.getInput(), rewriter));
      return success();
    }
    rewriter.replaceOp(op, emitElementwiseLoop(
                               op.getLoc(), resultType, adaptor.getInput(),
                               [&](ArrayRef<Value> elements, OpBuilder &builder) {
                                 return emitQuantizeBody(op, elements.front(), builder);
                               },
                               rewriter));
    return success();
  }
};

// The elementwise family lowers to one loop per operation whose body is the
// exact integer expression followed by the operation's single declared
// boundary. Widening to twice the storage first is what makes "exact" true of
// the body: nothing can lose a bit before the boundary the contract names.
template <typename SourceOp>
class ElementwiseOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    RankedTensorType resultType = op.getResult().getType();
    auto storage = cast<IntegerType>(resultType.getElementType());
    IntegerType wide = rewriter.getIntegerType(2 * storage.getWidth());

    SmallVector<Value> sources;
    for (Value operand : adaptor.getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        sources.push_back(operand);

    rewriter.replaceOp(op, emitElementwiseLoop(
                               loc, resultType, sources,
                               [&](ArrayRef<Value> elements, OpBuilder &builder) {
                                 return emitBody(op, elements, context, storage, wide, loc,
                                                 builder);
                               },
                               rewriter));
    return success();
  }

private:
  static ondrix::ondsp::ScaleAttr narrowingScale(MLIRContext *context, unsigned shift,
                                                 ondrix::ondsp::RoundingMode rounding,
                                                 ondrix::ondsp::OverflowMode overflow,
                                                 IntegerType destination) {
    return ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, shift, rounding, overflow,
                                         destination);
  }

  static Value emitBody(SourceOp op, ArrayRef<Value> elements, MLIRContext *context,
                        IntegerType storage, IntegerType wide, Location loc, OpBuilder &builder);
};

template <>
Value ElementwiseOpLowering<ondrix::ir::AddOp>::emitBody(ondrix::ir::AddOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType storage,
                                                         IntegerType wide, Location loc,
                                                         OpBuilder &builder) {
  // add_shift computes the sum one bit wider than its operands, so the
  // storage-width operands give the exact sum and the scale is the only
  // boundary.
  return builder.create<ondrix::ondsp::AddShiftOp>(
      loc, storage, elements[0], elements[1],
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::SubOp>::emitBody(ondrix::ir::SubOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType storage,
                                                         IntegerType wide, Location loc,
                                                         OpBuilder &builder) {
  return builder.create<ondrix::ondsp::SubShiftOp>(
      loc, storage, elements[0], elements[1],
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::MultOp>::emitBody(ondrix::ir::MultOp op,
                                                          ArrayRef<Value> elements,
                                                          MLIRContext *context, IntegerType storage,
                                                          IntegerType wide, Location loc,
                                                          OpBuilder &builder) {
  Value lhs = builder.create<arith::ExtSIOp>(loc, wide, elements[0]);
  Value rhs = builder.create<arith::ExtSIOp>(loc, wide, elements[1]);
  Value product = builder.create<arith::MulIOp>(loc, lhs, rhs);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, storage, product,
      narrowingScale(context, storage.getWidth() - 1, op.getRounding(), op.getOverflow(), storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::RatioOp>::emitBody(ondrix::ir::RatioOp op,
                                                           ArrayRef<Value> elements,
                                                           MLIRContext *context,
                                                           IntegerType storage, IntegerType wide,
                                                           Location loc, OpBuilder &builder) {
  // The quotient's own pre-scale widens inside the operation, so the body
  // is the one boundary on the storage values.
  return builder.create<ondrix::ondsp::RoundQuotientOp>(
      loc, storage, elements[0], elements[1], builder.getI64IntegerAttr(storage.getWidth() - 1),
      op.getRoundingAttr(), op.getOverflowAttr(), op.getNonpositiveAttr());
}

template <>
Value ElementwiseOpLowering<ondrix::ir::AbsOp>::emitBody(ondrix::ir::AbsOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType storage,
                                                         IntegerType wide, Location loc,
                                                         OpBuilder &builder) {
  // Negating one width up is what keeps the storage minimum's magnitude
  // exact; the declared overflow then decides the one input the destination
  // cannot hold.
  Value extended = builder.create<arith::ExtSIOp>(loc, wide, elements[0]);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, wide);
  Value negated = builder.create<arith::SubIOp>(loc, zero, extended);
  Value magnitude = builder.create<arith::MaxSIOp>(loc, extended, negated);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, storage, magnitude,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::NegateOp>::emitBody(ondrix::ir::NegateOp op,
                                                            ArrayRef<Value> elements,
                                                            MLIRContext *context,
                                                            IntegerType storage, IntegerType wide,
                                                            Location loc, OpBuilder &builder) {
  Value extended = builder.create<arith::ExtSIOp>(loc, wide, elements[0]);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, wide);
  Value negated = builder.create<arith::SubIOp>(loc, zero, extended);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, storage, negated,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::OffsetOp>::emitBody(ondrix::ir::OffsetOp op,
                                                            ArrayRef<Value> elements,
                                                            MLIRContext *context,
                                                            IntegerType storage, IntegerType wide,
                                                            Location loc, OpBuilder &builder) {
  Value bias = builder.create<arith::ConstantIntOp>(loc, op.getBiasAttr().getInt(), storage);
  return builder.create<ondrix::ondsp::AddShiftOp>(
      loc, storage, elements[0], bias,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::ShiftOp>::emitBody(ondrix::ir::ShiftOp op,
                                                           ArrayRef<Value> elements,
                                                           MLIRContext *context,
                                                           IntegerType storage, IntegerType wide,
                                                           Location loc, OpBuilder &builder) {
  int64_t amount = op.getAmountAttr().getInt();
  Value value = builder.create<arith::ExtSIOp>(loc, wide, elements[0]);
  if (amount > 0) {
    // Exact at twice the storage: the amount is bounded by the storage width
    // less one, so only the narrowing can lose.
    Value shift = builder.create<arith::ConstantIntOp>(loc, amount, wide);
    value = builder.create<arith::ShLIOp>(loc, value, shift);
  }
  unsigned right = amount < 0 ? unsigned(-amount) : 0u;
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, storage, value,
      narrowingScale(context, right, op.getRounding(), op.getOverflow(), storage));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::DivOp>::emitBody(ondrix::ir::DivOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType storage,
                                                         IntegerType wide, Location loc,
                                                         OpBuilder &builder) {
  // The quotient's magnitude never exceeds the input's, so the storage is its
  // own exact carrier: no pre-scale, and the narrowing is the declared boundary.
  return builder.create<ondrix::ondsp::RoundDivOp>(
      loc, storage, elements[0], builder.getI64IntegerAttr(op.getDivisorAttr().getInt()),
      builder.getI64IntegerAttr(0), ondrix::ondsp::RoundingModeAttr::get(context, op.getRounding()),
      ondrix::ondsp::OverflowModeAttr::get(context, op.getOverflow()));
}

// Shared table-plus-interpolation lowering for ondrix.sine/cosine. The
// phase offset is 0 for sine and 16384 (one exact quarter turn) for
// cosine; everything else — the tie-guarded 256-entry table, the Q8
// nearest-even interpolation boundary, and the saturating combine — is
// identical by contract.
static LogicalResult lowerQ15Trig(Operation *op, Value input, Value result, Attribute numeric,
                                  int64_t phaseOffset, ConversionPatternRewriter &rewriter) {
  // Compile-time table under the shared guarded quantizer; fail closed if
  // any entry were inadmissible (the committed profile is, by margin
  // evidence, but the guard stays as the backstop).
  SmallVector<int16_t> table;
  table.reserve(256);
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  for (int64_t k = 0; k < 256; ++k) {
    std::optional<ondrix::GuardedQ15Value> entry =
        ondrix::quantizeGuardedQ15(std::sin(kTwoPi * static_cast<double>(k) / 256.0));
    if (!entry)
      return rewriter.notifyMatchFailure(op, "sine table entry is not tie-guard admissible");
    table.push_back(entry->value);
  }

  Location loc = op->getLoc();
  IntegerType i16 = rewriter.getI16Type();
  IntegerType i32 = rewriter.getIntegerType(32);
  int64_t extent = cast<RankedTensorType>(input.getType()).getDimSize(0);
  auto interpolationScale = ondrix::ondsp::ScaleAttr::get(
      rewriter.getContext(), /*preShiftLeft=*/0, /*postShiftRight=*/8,
      ondrix::ondsp::RoundingMode::NearestEven, ondrix::ondsp::OverflowMode::Saturate, i16);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({256}, i16), llvm::ArrayRef<int16_t>(table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value offset = rewriter.create<arith::ConstantIntOp>(loc, phaseOffset, 32);
  Value phaseMask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFF, 32);
  Value indexShift = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
  Value fractionMask = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
  Value one32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{result},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        Value phase = builder.create<tensor::ExtractOp>(loc, input, position);
        // Zero extension reads the raw bits as the unsigned turn phase;
        // the offset add plus mask is the exact modular phase advance.
        Value raw = builder.create<arith::ExtUIOp>(loc, i32, phase);
        Value advanced = builder.create<arith::AddIOp>(loc, raw, offset);
        Value turn = builder.create<arith::AndIOp>(loc, advanced, phaseMask);
        Value tableIndex = builder.create<arith::ShRUIOp>(loc, turn, indexShift);
        Value fraction = builder.create<arith::AndIOp>(loc, turn, fractionMask);
        Value nextRaw = builder.create<arith::AddIOp>(loc, tableIndex, one32);
        Value nextIndex = builder.create<arith::AndIOp>(loc, nextRaw, fractionMask);
        Value lowerIdx =
            builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tableIndex);
        Value upperIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), nextIndex);
        Value lower = builder.create<tensor::ExtractOp>(loc, tableConstant, lowerIdx);
        Value upper = builder.create<tensor::ExtractOp>(loc, tableConstant, upperIdx);
        Value lowerWide = builder.create<arith::ExtSIOp>(loc, i32, lower);
        Value upperWide = builder.create<arith::ExtSIOp>(loc, i32, upper);
        Value delta = builder.create<arith::SubIOp>(loc, upperWide, lowerWide);
        Value product = builder.create<arith::MulIOp>(loc, delta, fraction);
        Value interpolated = builder.create<ondrix::ondsp::RoundShiftOp>(
            loc, builder.getI16Type(), product, interpolationScale);
        Value interpolatedWide = builder.create<arith::ExtSIOp>(loc, i32, interpolated);
        Value combined = builder.create<arith::AddIOp>(loc, lowerWide, interpolatedWide);
        Value saturated =
            builder.create<ondrix::ondsp::SatCastOp>(loc, builder.getI16Type(), combined, numeric);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, saturated, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

// The declared transcendental tables, generated under the same guarded
// quantizer the trigonometric table uses. Each has 129 entries so the
// interpolation's upper neighbour is a real entry rather than a wrapped one,
// and entry 128 is the exact declared endpoint (2048, 65536, or the eighth
// turn 8192), never a rounded approximation of it.
static std::optional<SmallVector<int32_t>>
buildDeclaredTable(int32_t exactEndpoint, llvm::function_ref<double(double)> exactValue) {
  SmallVector<int32_t> table;
  table.reserve(129);
  for (int64_t k = 0; k <= 128; ++k) {
    if (k == 128) {
      table.push_back(exactEndpoint);
      break;
    }
    double exact = exactValue(double(k) / 128.0);
    double rounded = std::nearbyint(exact);
    // The same tie guard the design tables use: an entry within 2^-20 of a
    // halfway point is not admissible evidence of which integer it is.
    if (std::abs(exact - rounded) > 0.5 - 9.5367431640625e-07)
      return std::nullopt;
    table.push_back(int32_t(rounded));
  }
  return table;
}

static std::optional<SmallVector<int32_t>> buildLog2Table() {
  return buildDeclaredTable(2048, [](double t) { return std::log2(1.0 + t) * 2048.0; });
}

static std::optional<SmallVector<int32_t>> buildExp2Table() {
  return buildDeclaredTable(65536, [](double t) { return std::exp2(t) * 32768.0; });
}

static std::optional<SmallVector<int32_t>> buildArctangentTable() {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  return buildDeclaredTable(8192, [](double t) { return std::atan(t) / kTwoPi * 65536.0; });
}

// The Q31-profile tables: 1024 coarse points read by a correction series
// rather than by interpolation, quantized under the Q31 tie guard at the scale
// each description names and stored as raw 32-bit words (the exp2 table reads
// back unsigned). An optional exact endpoint closes the arctangent's eighth turn.
static std::optional<SmallVector<int32_t>>
buildQ31Table(double scale, llvm::function_ref<double(double)> exactValue,
              std::optional<int64_t> exactEndpoint) {
  SmallVector<int32_t> table;
  table.reserve(1025);
  for (int64_t k = 0; k < 1024; ++k) {
    std::optional<int64_t> word = ondrix::quantizeGuardedAtScale(exactValue(k / 1024.0), scale);
    if (!word || *word < 0 || *word > 0xFFFFFFFFLL)
      return std::nullopt;
    table.push_back(static_cast<int32_t>(static_cast<uint32_t>(*word)));
  }
  if (exactEndpoint)
    table.push_back(static_cast<int32_t>(*exactEndpoint));
  return table;
}

static std::optional<SmallVector<int32_t>> buildLog2TableQ31() {
  return buildQ31Table(
      1073741824.0, [](double t) { return std::log2(1.0 + t); }, std::nullopt);
}

static std::optional<SmallVector<int32_t>> buildExp2TableQ31() {
  return buildQ31Table(
      2147483648.0, [](double t) { return std::exp2(t); }, std::nullopt);
}

static std::optional<SmallVector<int32_t>> buildArctangentTableQ32() {
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  return buildQ31Table(
      8589934592.0, [](double t) { return std::atan(t) / kTwoPi; }, int64_t(1) << 30);
}

// The Q0.32 turn: the ratio is carried to 2^-32 and the arctangent is the
// exact angle addition `atan(x) = atan(r_k) + atan(u)` with
// `u = (x - r_k) / (1 + x*r_k)` below 2^-10, so `atan(u) ~ u - u^3/3` needs
// one table. The component unpack is the Q0.16 path's; only the turn differs.
static LogicalResult lowerWideTurnCxPhase(ondrix::ir::CxPhaseOp op, Value input,
                                          ConversionPatternRewriter &rewriter) {
  std::optional<SmallVector<int32_t>> table = buildArctangentTableQ32();
  if (!table)
    return rewriter.notifyMatchFailure(op,
                                       "Q0.32 arctangent table entry is not tie-guard admissible");
  Location loc = op.getLoc();
  MLIRContext *context = rewriter.getContext();
  IntegerType i32 = rewriter.getIntegerType(32);
  IntegerType i64 = rewriter.getIntegerType(64);
  ondrix::ondsp::PackedComplexProfile profile =
      *ondrix::ondsp::getPackedComplexProfile(op.getLayout().getLayout());
  IntegerType component = rewriter.getIntegerType(profile.storageWidth);
  IntegerType container = rewriter.getIntegerType(profile.containerWidth);
  RankedTensorType resultType = op.getResult().getType();
  int64_t extent = resultType.getDimSize(0);
  auto turnScale =
      ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/33,
                                    op.getRounding(), ondrix::ondsp::OverflowMode::Saturate, i32);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({1025}, i32), llvm::ArrayRef<int32_t>(*table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i32);
  // 1/(2*pi) at 2^40, which returns the radian correction to turns.
  Value inverseTwoPi = rewriter.create<arith::ConstantIntOp>(loc, 174992710548LL, i64);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{empty},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        auto constant = [&](int64_t value) -> Value {
          return builder.create<arith::ConstantIntOp>(loc, value, i64);
        };
        Value packed = builder.create<tensor::ExtractOp>(loc, input, position);
        Value real = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<arith::TruncIOp>(loc, component, packed));
        Value imaginary = builder.create<arith::ExtSIOp>(
            loc, i64,
            builder.create<arith::TruncIOp>(
                loc, component,
                builder.create<arith::ShRSIOp>(
                    loc, packed,
                    builder.create<arith::ConstantIntOp>(loc, profile.storageWidth, container))));
        Value zero64 = constant(0);
        Value absReal = builder.create<arith::MaxSIOp>(
            loc, real, builder.create<arith::SubIOp>(loc, zero64, real));
        Value absImaginary = builder.create<arith::MaxSIOp>(
            loc, imaginary, builder.create<arith::SubIOp>(loc, zero64, imaginary));
        Value swapped =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, absImaginary, absReal);
        Value high = builder.create<arith::MaxSIOp>(loc, absReal, absImaginary);
        Value low = builder.create<arith::MinSIOp>(loc, absReal, absImaginary);
        Value atOrigin = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, high, zero64);
        Value divisor = builder.create<arith::SelectOp>(loc, atOrigin, constant(1), high);
        // The ratio at 2^-32, rounded once. Unsigned throughout: low reaches
        // 2^31 at the component minimum and its shifted numerator is 2^63.
        Value numerator = builder.create<arith::ShLIOp>(loc, low, constant(32));
        Value quotient = builder.create<arith::DivUIOp>(loc, numerator, divisor);
        Value remainder = builder.create<arith::SubIOp>(
            loc, numerator, builder.create<arith::MulIOp>(loc, quotient, divisor));
        Value doubled = builder.create<arith::AddIOp>(loc, remainder, remainder);
        Value aboveHalf =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, doubled, divisor);
        Value atHalf =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, doubled, divisor);
        Value odd = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne,
            builder.create<arith::AndIOp>(loc, quotient, constant(1)), zero64);
        Value stepUp = builder.create<arith::OrIOp>(
            loc, aboveHalf, builder.create<arith::AndIOp>(loc, atHalf, odd));
        Value ratio = builder.create<arith::AddIOp>(
            loc, quotient, builder.create<arith::SelectOp>(loc, stepUp, constant(1), zero64));
        // Coarse point k = x at 2^-10 (1024 when the ratio is exactly one),
        // residual d below it, then u = d / (1 + x*r_k) at 2^-32.
        Value coarse = builder.create<arith::ShRUIOp>(loc, ratio, constant(22));
        Value residual = builder.create<arith::SubIOp>(
            loc, ratio, builder.create<arith::ShLIOp>(loc, coarse, constant(22)));
        Value denominator = builder.create<arith::AddIOp>(
            loc, constant(int64_t(1) << 42), builder.create<arith::MulIOp>(loc, ratio, coarse));
        Value u = builder.create<arith::DivUIOp>(
            loc, builder.create<arith::ShLIOp>(loc, residual, constant(42)), denominator);
        Value uSquare = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, u, u), constant(22));
        Value uCube = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, uSquare, u), constant(22));
        Value linear = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, u, inverseTwoPi), constant(7));
        Value cubic = builder.create<arith::ShRUIOp>(
            loc,
            builder.create<arith::DivUIOp>(
                loc, builder.create<arith::MulIOp>(loc, uCube, inverseTwoPi), constant(3)),
            constant(27));
        Value correction = builder.create<arith::SubIOp>(loc, linear, cubic);
        Value coarseIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), coarse);
        Value tableWord = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<tensor::ExtractOp>(loc, tableConstant, coarseIdx));
        Value total = builder.create<arith::AddIOp>(
            loc, builder.create<arith::ShLIOp>(loc, tableWord, constant(32)), correction);
        Value base = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<ondrix::ondsp::RoundShiftOp>(loc, i32, total, turnScale));
        // Exact turn arithmetic from here: the octant fold and quadrant
        // unfold, then the truncation is the modulo-2^32 turn.
        Value folded = builder.create<arith::SelectOp>(
            loc, swapped, builder.create<arith::SubIOp>(loc, constant(int64_t(1) << 30), base),
            base);
        Value nonNegativeReal =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, real, zero64);
        Value nonNegativeImaginary =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, imaginary, zero64);
        Value half = constant(int64_t(1) << 31);
        Value right = builder.create<arith::SelectOp>(
            loc, nonNegativeImaginary, folded, builder.create<arith::SubIOp>(loc, zero64, folded));
        Value left = builder.create<arith::SelectOp>(
            loc, nonNegativeImaginary, builder.create<arith::SubIOp>(loc, half, folded),
            builder.create<arith::AddIOp>(loc, half, folded));
        Value turn = builder.create<arith::SelectOp>(loc, nonNegativeReal, right, left);
        Value selected = builder.create<arith::SelectOp>(loc, atOrigin, zero64, turn);
        Value narrowed = builder.create<arith::TruncIOp>(loc, i32, selected);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, narrowed, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

class CxPhaseOpLowering final : public OpConversionPattern<ondrix::ir::CxPhaseOp> {
public:
  using OpConversionPattern<ondrix::ir::CxPhaseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::CxPhaseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // The ratio division below writes nearest-even inline (its divisor is a
    // runtime value, so round_div cannot carry it). Same self-guard as exp2:
    // one operation must not follow two tie rules without a diagnostic.
    if (op.getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return rewriter.notifyMatchFailure(op, "cx_phase lowering implements nearest_even only");
    if (op.getOutputNumeric().getStorage().isSignlessInteger(32))
      return lowerWideTurnCxPhase(op, adaptor.getInput(), rewriter);
    std::optional<SmallVector<int32_t>> table = buildArctangentTable();
    if (!table)
      return rewriter.notifyMatchFailure(op, "arctangent table entry is not tie-guard admissible");

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
    // Only the component arithmetic follows the declared width. The ratio, the
    // table, the interpolation and the octant fold all live on the Q0.16 turn,
    // which is the reading `ondrix.sine` consumes at either width.
    ondrix::ondsp::PackedComplexProfile profile =
        *ondrix::ondsp::getPackedComplexProfile(op.getLayout().getLayout());
    unsigned componentWidth = profile.storageWidth;
    IntegerType component = rewriter.getIntegerType(componentWidth);
    IntegerType container = rewriter.getIntegerType(profile.containerWidth);
    IntegerType work = rewriter.getIntegerType(profile.containerWidth);
    RankedTensorType resultType = op.getResult().getType();
    int64_t extent = resultType.getDimSize(0);
    auto interpolation =
        ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/9,
                                      op.getRounding(), ondrix::ondsp::OverflowMode::Saturate, i32);

    Value tableConstant = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(RankedTensorType::get({129}, i32), llvm::ArrayRef<int32_t>(*table)));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i16);

    auto loop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          auto constant = [&](int64_t value, IntegerType type) -> Value {
            return builder.create<arith::ConstantIntOp>(loc, value, type);
          };
          Value packed = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value real = builder.create<arith::ExtSIOp>(
              loc, work, builder.create<arith::TruncIOp>(loc, component, packed));
          Value imaginary = builder.create<arith::ExtSIOp>(
              loc, work,
              builder.create<arith::TruncIOp>(
                  loc, component,
                  builder.create<arith::ShRSIOp>(loc, packed,
                                                 constant(componentWidth, container))));
          // Two zero constants, and the split is load bearing: the component
          // arithmetic runs one bit wider than the component (negating the
          // component minimum is part of taking the absolute value) while the
          // turn arithmetic below stays on the Q0.16 reading at either width.
          Value zeroWork = constant(0, work);
          Value oneWork = constant(1, work);
          Value zero32 = constant(0, i32);
          Value one32 = constant(1, i32);
          Value absReal = builder.create<arith::MaxSIOp>(
              loc, real, builder.create<arith::SubIOp>(loc, zeroWork, real));
          Value absImaginary = builder.create<arith::MaxSIOp>(
              loc, imaginary, builder.create<arith::SubIOp>(loc, zeroWork, imaginary));
          Value swapped =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, absImaginary, absReal);
          Value high = builder.create<arith::MaxSIOp>(loc, absReal, absImaginary);
          Value low = builder.create<arith::MinSIOp>(loc, absReal, absImaginary);
          // The origin has no argument; the divisor is forced to one so the
          // division is defined, and the declared value replaces the result.
          Value atOrigin =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, high, zeroWork);
          Value divisor = builder.create<arith::SelectOp>(loc, atOrigin, oneWork, high);

          // The ratio, rounded once. Both operands are non-negative, so the
          // truncating division is the floor and the remainder is the
          // Euclidean one; the tie test compares 2*remainder against the
          // divisor rather than forming a half that may not be an integer.
          Value wideLow = work == i64 ? low : builder.create<arith::ExtSIOp>(loc, i64, low);
          Value numerator = builder.create<arith::ShLIOp>(loc, wideLow, constant(16, i64));
          Value wideDivisor =
              work == i64 ? divisor : builder.create<arith::ExtSIOp>(loc, i64, divisor).getResult();
          Value quotient = builder.create<arith::DivSIOp>(loc, numerator, wideDivisor);
          Value remainder = builder.create<arith::SubIOp>(
              loc, numerator, builder.create<arith::MulIOp>(loc, quotient, wideDivisor));
          Value doubled = builder.create<arith::AddIOp>(loc, remainder, remainder);
          Value aboveHalf =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, doubled, wideDivisor);
          Value atHalf =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, doubled, wideDivisor);
          Value odd = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne,
              builder.create<arith::AndIOp>(loc, quotient, constant(1, i64)), constant(0, i64));
          Value stepUp = builder.create<arith::OrIOp>(
              loc, aboveHalf, builder.create<arith::AndIOp>(loc, atHalf, odd));
          Value ratio = builder.create<arith::TruncIOp>(
              loc, i32,
              builder.create<arith::AddIOp>(loc, quotient,
                                            builder.create<arith::SelectOp>(
                                                loc, stepUp, constant(1, i64), constant(0, i64))));

          Value rawIndex = builder.create<arith::ShRUIOp>(loc, ratio, constant(9, i32));
          Value tableIndex = builder.create<arith::MinSIOp>(loc, rawIndex, constant(127, i32));
          Value fraction = builder.create<arith::SubIOp>(
              loc, ratio, builder.create<arith::ShLIOp>(loc, tableIndex, constant(9, i32)));
          Value lowerIdx =
              builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tableIndex);
          Value upperIdx = builder.create<arith::IndexCastOp>(
              loc, builder.getIndexType(), builder.create<arith::AddIOp>(loc, tableIndex, one32));
          Value lower = builder.create<tensor::ExtractOp>(loc, tableConstant, lowerIdx);
          Value upper = builder.create<tensor::ExtractOp>(loc, tableConstant, upperIdx);
          Value interpolated = builder.create<ondrix::ondsp::RoundShiftOp>(
              loc, i32,
              builder.create<arith::MulIOp>(loc, builder.create<arith::SubIOp>(loc, upper, lower),
                                            fraction),
              interpolation);
          Value base = builder.create<arith::AddIOp>(loc, lower, interpolated);

          // From here everything is exact turn arithmetic, which is what
          // makes the octant boundaries meet rather than nearly meet.
          Value folded = builder.create<arith::SelectOp>(
              loc, swapped, builder.create<arith::SubIOp>(loc, constant(16384, i32), base), base);
          Value nonNegativeReal =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, real, zeroWork);
          Value nonNegativeImaginary =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, imaginary, zeroWork);
          Value half = constant(32768, i32);
          Value right =
              builder.create<arith::SelectOp>(loc, nonNegativeImaginary, folded,
                                              builder.create<arith::SubIOp>(loc, zero32, folded));
          Value left = builder.create<arith::SelectOp>(
              loc, nonNegativeImaginary, builder.create<arith::SubIOp>(loc, half, folded),
              builder.create<arith::AddIOp>(loc, half, folded));
          Value turn = builder.create<arith::SelectOp>(loc, nonNegativeReal, right, left);
          Value selected = builder.create<arith::SelectOp>(loc, atOrigin, zero32, turn);
          Value narrowed = builder.create<arith::TruncIOp>(loc, i16, selected);
          Value inserted =
              builder.create<tensor::InsertOp>(loc, narrowed, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

// The Q31 logarithm: exponent plus a 1024-point Q30 table plus the series
// log2(1 + r) ~ (r - r^2/2) / ln 2 for the residual ratio r below 2^-10, which
// one 64-bit division supplies. The dropped cubic term is under 2^-31 of a
// unit, and the only rounding boundary is the return to Q6.26.
static LogicalResult lowerQ31Log2(ondrix::ir::Log2Op op, Value input,
                                  ConversionPatternRewriter &rewriter) {
  std::optional<SmallVector<int32_t>> table = buildLog2TableQ31();
  if (!table)
    return rewriter.notifyMatchFailure(op, "Q31 log2 table entry is not tie-guard admissible");
  Location loc = op.getLoc();
  MLIRContext *context = rewriter.getContext();
  IntegerType i32 = rewriter.getIntegerType(32);
  IntegerType i64 = rewriter.getIntegerType(64);
  RankedTensorType resultType = op.getResult().getType();
  int64_t extent = resultType.getDimSize(0);
  auto mantissaScale =
      ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/30,
                                    op.getRounding(), ondrix::ondsp::OverflowMode::Saturate, i32);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({1024}, i32), llvm::ArrayRef<int32_t>(*table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i32);
  // 1/ln2 at 2^24 for the linear term and at 2^13 for the halved square, so
  // both land at 2^56 alongside the table word.
  Value linearScale = rewriter.create<arith::ConstantIntOp>(loc, 24204406, i64);
  Value squareScale = rewriter.create<arith::ConstantIntOp>(loc, 11819, i64);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{empty},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        auto constant32 = [&](int64_t value) -> Value {
          return builder.create<arith::ConstantIntOp>(loc, value, i32);
        };
        auto constant64 = [&](int64_t value) -> Value {
          return builder.create<arith::ConstantIntOp>(loc, value, i64);
        };
        Value magnitude = builder.create<tensor::ExtractOp>(loc, input, position);
        Value isPole =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, magnitude, constant32(0));
        // The pole takes the declared value; its arithmetic path is fed a one
        // so no shift amount reaches the width.
        Value safe = builder.create<arith::SelectOp>(loc, isPole, constant32(1), magnitude);
        Value exponent = builder.create<arith::SubIOp>(
            loc, constant32(31), builder.create<math::CountLeadingZerosOp>(loc, safe));
        Value mantissa = builder.create<arith::ExtUIOp>(
            loc, i64,
            builder.create<arith::ShLIOp>(
                loc, safe, builder.create<arith::SubIOp>(loc, constant32(31), exponent)));
        Value coarse = builder.create<arith::ShRUIOp>(loc, mantissa, constant64(21));
        Value base = builder.create<arith::ShLIOp>(loc, coarse, constant64(21));
        Value residual = builder.create<arith::SubIOp>(loc, mantissa, base);
        Value ratio = builder.create<arith::DivUIOp>(
            loc, builder.create<arith::ShLIOp>(loc, residual, constant64(32)), base);
        Value ratioSquare = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, ratio, ratio), constant64(22));
        Value series = builder.create<arith::SubIOp>(
            loc, builder.create<arith::MulIOp>(loc, ratio, linearScale),
            builder.create<arith::MulIOp>(loc, ratioSquare, squareScale));
        Value tableIdx = builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(),
            builder.create<arith::SubIOp>(loc, coarse, constant64(1024)));
        Value tableWord = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<tensor::ExtractOp>(loc, tableConstant, tableIdx));
        Value total = builder.create<arith::AddIOp>(
            loc, builder.create<arith::ShLIOp>(loc, tableWord, constant64(26)), series);
        Value fraction =
            builder.create<ondrix::ondsp::RoundShiftOp>(loc, i32, total, mantissaScale);
        Value binade = builder.create<arith::ShLIOp>(
            loc, builder.create<arith::SubIOp>(loc, exponent, constant32(32)), constant32(26));
        Value sum = builder.create<arith::AddIOp>(loc, binade, fraction);
        Value selected = builder.create<arith::SelectOp>(loc, isPole, constant32(INT32_MIN), sum);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, selected, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

// The Q31 exponential: a 1024-point table of 2^(k/1024) at 2^31 read back
// unsigned, the series 2^x - 1 ~ a + a^2/2 + a^3/6 with a = x ln 2 below
// 2^-10.5, and the binade placement taken on the 2^62-scaled product so the
// whole element has ONE rounding boundary where the Q15 profile needs two.
static LogicalResult lowerQ31Exp2(ondrix::ir::Exp2Op op, Value input,
                                  ConversionPatternRewriter &rewriter) {
  std::optional<SmallVector<int32_t>> table = buildExp2TableQ31();
  if (!table)
    return rewriter.notifyMatchFailure(op, "Q31 exp2 table entry is not tie-guard admissible");
  Location loc = op.getLoc();
  IntegerType i32 = rewriter.getIntegerType(32);
  IntegerType i64 = rewriter.getIntegerType(64);
  RankedTensorType resultType = op.getResult().getType();
  int64_t extent = resultType.getDimSize(0);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({1024}, i32), llvm::ArrayRef<int32_t>(*table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i32);
  Value logTwo = rewriter.create<arith::ConstantIntOp>(loc, 762123384786LL, i64); // ln 2 at 2^40

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{empty},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        auto constant32 = [&](int64_t value) -> Value {
          return builder.create<arith::ConstantIntOp>(loc, value, i32);
        };
        auto constant64 = [&](int64_t value) -> Value {
          return builder.create<arith::ConstantIntOp>(loc, value, i64);
        };
        Value value = builder.create<tensor::ExtractOp>(loc, input, position);
        Value aboveRange =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, value, constant32(0));
        Value exponent = builder.create<arith::ShRSIOp>(loc, value, constant32(26));
        Value fraction = builder.create<arith::AndIOp>(loc, value, constant32((1 << 26) - 1));
        Value coarseIdx = builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(),
            builder.create<arith::ShRUIOp>(loc, fraction, constant32(16)));
        Value residual = builder.create<arith::ExtUIOp>(
            loc, i64, builder.create<arith::AndIOp>(loc, fraction, constant32(0xFFFF)));
        // a = x ln2 at 2^40 and its two powers, then 2^x - 1 at 2^40.
        Value angle = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, residual, logTwo), constant64(26));
        Value square = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, angle, angle), constant64(40));
        Value cube = builder.create<arith::ShRUIOp>(
            loc, builder.create<arith::MulIOp>(loc, square, angle), constant64(40));
        Value series = builder.create<arith::AddIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, angle, builder.create<arith::ShRUIOp>(loc, square, constant64(1))),
            builder.create<arith::DivUIOp>(loc, cube, constant64(6)));
        Value tableWord = builder.create<arith::ExtUIOp>(
            loc, i64, builder.create<tensor::ExtractOp>(loc, tableConstant, coarseIdx));
        // The mantissa at 2^62: table word at 2^31 plus its scaled series.
        Value wide = builder.create<arith::AddIOp>(
            loc, builder.create<arith::ShLIOp>(loc, tableWord, constant64(31)),
            builder.create<arith::ShRUIOp>(
                loc, builder.create<arith::MulIOp>(loc, tableWord, series), constant64(9)));
        // The binade placement is the one boundary, written out because its
        // amount is input-dependent; above the range the amount is pinned
        // away from an undefined shift and the ceiling is selected instead.
        Value places = builder.create<arith::ExtSIOp>(
            loc, i64,
            builder.create<arith::SelectOp>(
                loc, aboveRange, constant32(31),
                builder.create<arith::SubIOp>(loc, constant32(30), exponent)));
        Value quotient = builder.create<arith::ShRUIOp>(loc, wide, places);
        Value remainder = builder.create<arith::SubIOp>(
            loc, wide, builder.create<arith::ShLIOp>(loc, quotient, places));
        Value half = builder.create<arith::ShLIOp>(
            loc, constant64(1), builder.create<arith::SubIOp>(loc, places, constant64(1)));
        Value aboveHalf =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remainder, half);
        Value atHalf =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, half);
        Value odd = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne,
            builder.create<arith::AndIOp>(loc, quotient, constant64(1)), constant64(0));
        Value stepUp = builder.create<arith::OrIOp>(
            loc, aboveHalf, builder.create<arith::AndIOp>(loc, atHalf, odd));
        Value rounded = builder.create<arith::AddIOp>(
            loc, quotient,
            builder.create<arith::SelectOp>(loc, stepUp, constant64(1), constant64(0)));
        Value selected =
            builder.create<arith::SelectOp>(loc, aboveRange, constant64(0xFFFFFFFFLL), rounded);
        Value narrowed = builder.create<arith::TruncIOp>(loc, i32, selected);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, narrowed, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

class Log2OpLowering final : public OpConversionPattern<ondrix::ir::Log2Op> {
public:
  using OpConversionPattern<ondrix::ir::Log2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::Log2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumeric().getStorage().isSignlessInteger(32))
      return lowerQ31Log2(op, adaptor.getInput(), rewriter);
    std::optional<SmallVector<int32_t>> table = buildLog2Table();
    if (!table)
      return rewriter.notifyMatchFailure(op, "log2 table entry is not tie-guard admissible");

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    RankedTensorType resultType = op.getResult().getType();
    int64_t extent = resultType.getDimSize(0);
    auto interpolation =
        ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/8,
                                      op.getRounding(), ondrix::ondsp::OverflowMode::Saturate, i32);

    Value tableConstant = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(RankedTensorType::get({129}, i32), llvm::ArrayRef<int32_t>(*table)));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, i32);
    Value one32 = rewriter.create<arith::ConstantIntOp>(loc, 1, i32);
    Value fifteen = rewriter.create<arith::ConstantIntOp>(loc, 15, i32);
    Value eight = rewriter.create<arith::ConstantIntOp>(loc, 8, i32);
    Value indexMask = rewriter.create<arith::ConstantIntOp>(loc, 127, i32);
    Value fractionMask = rewriter.create<arith::ConstantIntOp>(loc, 255, i32);
    Value scale = rewriter.create<arith::ConstantIntOp>(loc, 2048, i32);
    Value bias = rewriter.create<arith::ConstantIntOp>(loc, 16, i32);
    Value pole = rewriter.create<arith::ConstantIntOp>(loc, -32768, i32);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i16);

    auto loop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          Value element = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value magnitude = builder.create<arith::ExtUIOp>(loc, i32, element);
          // The exponent is the index of the highest set bit; the count of
          // leading zeros gives it directly and is defined at every nonzero
          // input, which the pole branch handles separately.
          Value leading = builder.create<math::CountLeadingZerosOp>(loc, magnitude);
          Value thirtyOne = builder.create<arith::ConstantIntOp>(loc, 31, i32);
          Value exponent = builder.create<arith::SubIOp>(loc, thirtyOne, leading);
          Value shift = builder.create<arith::SubIOp>(loc, fifteen, exponent);
          Value mantissa = builder.create<arith::ShLIOp>(loc, magnitude, shift);
          Value tableIndex = builder.create<arith::AndIOp>(
              loc, builder.create<arith::ShRUIOp>(loc, mantissa, eight), indexMask);
          Value fraction = builder.create<arith::AndIOp>(loc, mantissa, fractionMask);
          Value lowerIdx =
              builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tableIndex);
          Value upperIdx = builder.create<arith::IndexCastOp>(
              loc, builder.getIndexType(), builder.create<arith::AddIOp>(loc, tableIndex, one32));
          Value lower = builder.create<tensor::ExtractOp>(loc, tableConstant, lowerIdx);
          Value upper = builder.create<tensor::ExtractOp>(loc, tableConstant, upperIdx);
          Value delta = builder.create<arith::SubIOp>(loc, upper, lower);
          Value product = builder.create<arith::MulIOp>(loc, delta, fraction);
          Value interpolated =
              builder.create<ondrix::ondsp::RoundShiftOp>(loc, i32, product, interpolation);
          Value binade = builder.create<arith::MulIOp>(
              loc, builder.create<arith::SubIOp>(loc, exponent, bias), scale);
          Value sum = builder.create<arith::AddIOp>(
              loc, binade, builder.create<arith::AddIOp>(loc, lower, interpolated));
          Value isPole =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, magnitude, zero32);
          Value selected = builder.create<arith::SelectOp>(loc, isPole, pole, sum);
          Value narrowed = builder.create<arith::TruncIOp>(loc, i16, selected);
          Value inserted =
              builder.create<tensor::InsertOp>(loc, narrowed, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class Exp2OpLowering final : public OpConversionPattern<ondrix::ir::Exp2Op> {
public:
  using OpConversionPattern<ondrix::ir::Exp2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::Exp2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // The binade placement below writes nearest-even inline (its shift
    // amount is input-dependent, so round_shift cannot carry it). The
    // verifier pins the mode; this guard keeps the lowering from silently
    // applying two different rules if that pin is ever widened.
    if (op.getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return rewriter.notifyMatchFailure(op, "exp2 lowering implements nearest_even only");
    if (op.getNumeric().getStorage().isSignlessInteger(32))
      return lowerQ31Exp2(op, adaptor.getInput(), rewriter);
    std::optional<SmallVector<int32_t>> table = buildExp2Table();
    if (!table)
      return rewriter.notifyMatchFailure(op, "exp2 table entry is not tie-guard admissible");

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    RankedTensorType resultType = op.getResult().getType();
    int64_t extent = resultType.getDimSize(0);
    auto interpolation =
        ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/4,
                                      op.getRounding(), ondrix::ondsp::OverflowMode::Saturate, i32);

    Value tableConstant = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(RankedTensorType::get({129}, i32), llvm::ArrayRef<int32_t>(*table)));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, i32);
    Value one32 = rewriter.create<arith::ConstantIntOp>(loc, 1, i32);
    Value four = rewriter.create<arith::ConstantIntOp>(loc, 4, i32);
    Value eleven = rewriter.create<arith::ConstantIntOp>(loc, 11, i32);
    Value fractionMask = rewriter.create<arith::ConstantIntOp>(loc, 2047, i32);
    Value indexMask = rewriter.create<arith::ConstantIntOp>(loc, 15, i32);
    Value minusOne = rewriter.create<arith::ConstantIntOp>(loc, -1, i32);
    Value ceiling = rewriter.create<arith::ConstantIntOp>(loc, 65535, i32);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i16);

    auto loop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          Value element = builder.create<tensor::ExtractOp>(loc, adaptor.getInput(), position);
          Value value = builder.create<arith::ExtSIOp>(loc, i32, element);
          // Arithmetic shift and mask split the Q5.11 value into a floor
          // exponent and a non-negative fraction at every input, including
          // the negative ones the range is made of.
          Value exponent = builder.create<arith::ShRSIOp>(loc, value, eleven);
          Value fraction = builder.create<arith::AndIOp>(loc, value, fractionMask);
          Value tableIndex = builder.create<arith::ShRUIOp>(loc, fraction, four);
          Value interpolant = builder.create<arith::AndIOp>(loc, fraction, indexMask);
          Value lowerIdx =
              builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tableIndex);
          Value upperIdx = builder.create<arith::IndexCastOp>(
              loc, builder.getIndexType(), builder.create<arith::AddIOp>(loc, tableIndex, one32));
          Value lower = builder.create<tensor::ExtractOp>(loc, tableConstant, lowerIdx);
          Value upper = builder.create<tensor::ExtractOp>(loc, tableConstant, upperIdx);
          Value delta = builder.create<arith::SubIOp>(loc, upper, lower);
          Value product = builder.create<arith::MulIOp>(loc, delta, interpolant);
          Value interpolated =
              builder.create<ondrix::ondsp::RoundShiftOp>(loc, i32, product, interpolation);
          Value mantissa = builder.create<arith::AddIOp>(loc, lower, interpolated);
          // The binade placement is a shift by an input-dependent amount, so
          // it is written out rather than expressed as a round_shift, whose
          // amount is part of its attribute.
          Value places = builder.create<arith::SubIOp>(loc, minusOne, exponent);
          // The top binade shifts by zero, where the half a nearest rule
          // compares against does not exist; the amount is clamped away from
          // an undefined shift and the whole rounding is selected out.
          Value shifting =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, places, zero32);
          Value safePlaces = builder.create<arith::MaxSIOp>(loc, places, one32);
          Value half = builder.create<arith::ShLIOp>(
              loc, one32, builder.create<arith::SubIOp>(loc, safePlaces, one32));
          Value quotient = builder.create<arith::ShRSIOp>(loc, mantissa, safePlaces);
          Value remainder = builder.create<arith::SubIOp>(
              loc, mantissa, builder.create<arith::ShLIOp>(loc, quotient, safePlaces));
          Value aboveHalf =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, remainder, half);
          Value atHalf =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, half);
          Value odd = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne, builder.create<arith::AndIOp>(loc, quotient, one32),
              zero32);
          Value stepUp = builder.create<arith::OrIOp>(
              loc, aboveHalf, builder.create<arith::AndIOp>(loc, atHalf, odd));
          Value shifted = builder.create<arith::AddIOp>(
              loc, quotient, builder.create<arith::SelectOp>(loc, stepUp, one32, zero32));
          Value rounded = builder.create<arith::SelectOp>(loc, shifting, shifted, mantissa);
          Value aboveRange =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, value, zero32);
          Value selected = builder.create<arith::SelectOp>(loc, aboveRange, ceiling, rounded);
          Value narrowed = builder.create<arith::TruncIOp>(loc, i16, selected);
          Value inserted =
              builder.create<tensor::InsertOp>(loc, narrowed, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

// The Q31 trigonometric contract. Widening the Q15 shape would not have
// worked: a 256-entry table read by linear interpolation sits about 2^-15.7
// from the real sine, so its storage can widen without its VALUE widening.
// This is instead the third-order angle-addition form
// `sin(C + A) = sin(C)cos(A) + cos(C)sin(A)` with `cos A ~ 1 - A^2/2` and
// `sin A ~ A - A^3/6`, over a 1024-entry table. One table serves both terms,
// because the cosine of a table angle is the sine a quarter turn on.
static LogicalResult lowerQ31Trig(Operation *op, Value input, Value result, Attribute numeric,
                                  int64_t phaseOffset, ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> table;
  table.reserve(1024);
  constexpr double kTwoPi = 6.28318530717958647692528676655900577;
  for (int64_t k = 0; k < 1024; ++k) {
    std::optional<ondrix::GuardedQ31Value> entry =
        ondrix::quantizeGuardedQ31(std::sin(kTwoPi * static_cast<double>(k) / 1024.0));
    if (!entry)
      return rewriter.notifyMatchFailure(op, "Q31 sine table entry is not tie-guard admissible");
    table.push_back(entry->value);
  }

  Location loc = op->getLoc();
  IntegerType i32 = rewriter.getIntegerType(32);
  IntegerType i64 = rewriter.getIntegerType(64);
  int64_t extent = cast<RankedTensorType>(input.getType()).getDimSize(0);
  // The one declared boundary: the correction returns to Q1.31 under the
  // op's pinned tie rule, and the combine below carries the saturation.
  auto correctionScale = ondrix::ondsp::ScaleAttr::get(
      rewriter.getContext(), /*preShiftLeft=*/0, /*postShiftRight=*/31,
      ondrix::ondsp::RoundingMode::NearestEven, ondrix::ondsp::OverflowMode::Saturate, i32);
  Value tableConstant = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({1024}, i32), llvm::ArrayRef<int32_t>(table)));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
  // 2*pi at 2^38: the largest scaling whose product with a 22-bit residual
  // stays inside i64, and 1.7e-13 relative, far under the value's own LSB.
  Value twoPi = rewriter.create<arith::ConstantIntOp>(loc, 1727108826179LL, i64);
  Value offset = rewriter.create<arith::ConstantIntOp>(loc, phaseOffset, i64);
  Value turnMask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFFFFFFLL, i64);
  Value indexShift = rewriter.create<arith::ConstantIntOp>(loc, 22, i64);
  Value residualMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3FFFFF, i64);
  Value quarter = rewriter.create<arith::ConstantIntOp>(loc, 256, i64);
  Value tableMask = rewriter.create<arith::ConstantIntOp>(loc, 1023, i64);
  Value angleShift = rewriter.create<arith::ConstantIntOp>(loc, 38, i64);
  Value squareShift = rewriter.create<arith::ConstantIntOp>(loc, 32, i64);
  Value oneBit = rewriter.create<arith::ConstantIntOp>(loc, 1, i64);
  Value twoBits = rewriter.create<arith::ConstantIntOp>(loc, 2, i64);
  Value twelve = rewriter.create<arith::ConstantIntOp>(loc, 12, i64);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, extentValue, one, ValueRange{result},
      [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
        Value phase = builder.create<tensor::ExtractOp>(loc, input, position);
        // Zero extension reads the raw bits as the unsigned turn phase; the
        // offset add plus mask is the exact modular phase advance.
        Value raw = builder.create<arith::ExtUIOp>(loc, i64, phase);
        Value advanced = builder.create<arith::AddIOp>(loc, raw, offset);
        Value turn = builder.create<arith::AndIOp>(loc, advanced, turnMask);
        Value coarse = builder.create<arith::ShRUIOp>(loc, turn, indexShift);
        Value residual = builder.create<arith::AndIOp>(loc, turn, residualMask);
        Value cosineRaw = builder.create<arith::AddIOp>(loc, coarse, quarter);
        Value cosineIndex = builder.create<arith::AndIOp>(loc, cosineRaw, tableMask);
        Value sineIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), coarse);
        Value cosineIdx =
            builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), cosineIndex);
        Value sine = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<tensor::ExtractOp>(loc, tableConstant, sineIdx));
        Value cosine = builder.create<arith::ExtSIOp>(
            loc, i64, builder.create<tensor::ExtractOp>(loc, tableConstant, cosineIdx));
        // The residual angle and its two powers, each one scaling down so the
        // next product stays inside i64.
        Value scaled = builder.create<arith::MulIOp>(loc, twoPi, residual);
        Value angle = builder.create<arith::ShRSIOp>(loc, scaled, angleShift);
        Value square = builder.create<arith::ShRSIOp>(
            loc, builder.create<arith::MulIOp>(loc, angle, angle), squareShift);
        Value cube = builder.create<arith::ShRSIOp>(
            loc, builder.create<arith::MulIOp>(loc, square, angle), squareShift);
        Value linear = builder.create<arith::ShRSIOp>(
            loc, builder.create<arith::MulIOp>(loc, cosine, angle), oneBit);
        Value quadratic = builder.create<arith::ShRSIOp>(
            loc, builder.create<arith::MulIOp>(loc, sine, square), twoBits);
        Value cubic = builder.create<arith::DivSIOp>(
            loc, builder.create<arith::MulIOp>(loc, cosine, cube), twelve);
        Value correction = builder.create<arith::SubIOp>(
            loc, builder.create<arith::SubIOp>(loc, linear, quadratic), cubic);
        Value narrowed =
            builder.create<ondrix::ondsp::RoundShiftOp>(loc, i32, correction, correctionScale);
        Value combined = builder.create<arith::AddIOp>(
            loc, sine, builder.create<arith::ExtSIOp>(loc, i64, narrowed));
        Value saturated = builder.create<ondrix::ondsp::SatCastOp>(loc, i32, combined, numeric);
        Value inserted =
            builder.create<tensor::InsertOp>(loc, saturated, iterArgs.front(), position);
        builder.create<scf::YieldOp>(loc, inserted);
      });
  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

// The phase offset is a quarter TURN, so it follows the phase width rather
// than being one constant: 2^14 at Q0.16 and 2^30 at Q0.32.
template <typename TrigOp, int64_t QuarterTurns>
class TrigOpLowering final : public OpConversionPattern<TrigOp> {
public:
  using OpConversionPattern<TrigOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TrigOp op, typename TrigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getResult().getType();
    unsigned storageWidth = resultType.getElementTypeBitWidth();
    Value empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(),
                                                   resultType.getElementType());
    int64_t offset = QuarterTurns * (int64_t(1) << (storageWidth - 2));
    if (storageWidth == 32)
      return lowerQ31Trig(op, adaptor.getInput(), empty, op.getNumeric(), offset, rewriter);
    return lowerQ15Trig(op, adaptor.getInput(), empty, op.getNumeric(), offset, rewriter);
  }
};

using SineOpLowering = TrigOpLowering<ondrix::ir::SineOp, 0>;
using CosineOpLowering = TrigOpLowering<ondrix::ir::CosineOp, 1>;

} // namespace

void ondrix::conversion::populateOndrixElementwiseLoweringPatterns(RewritePatternSet &patterns) {
  patterns
      .add<QuantizeOpLowering, SineOpLowering, CosineOpLowering, Log2OpLowering, CxPhaseOpLowering,
           Exp2OpLowering, ElementwiseOpLowering<ondrix::ir::AddOp>,
           ElementwiseOpLowering<ondrix::ir::SubOp>, ElementwiseOpLowering<ondrix::ir::MultOp>,
           ElementwiseOpLowering<ondrix::ir::AbsOp>, ElementwiseOpLowering<ondrix::ir::NegateOp>,
           ElementwiseOpLowering<ondrix::ir::OffsetOp>, ElementwiseOpLowering<ondrix::ir::ShiftOp>,
           ElementwiseOpLowering<ondrix::ir::DivOp>, ElementwiseOpLowering<ondrix::ir::RatioOp>>(
          patterns.getContext());
}
