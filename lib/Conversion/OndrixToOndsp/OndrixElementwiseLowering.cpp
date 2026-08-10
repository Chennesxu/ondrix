#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/GuardedQ15Quantization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

using namespace mlir;
using namespace ondrix::conversion;

namespace {

class QuantizeOpLowering final : public OpConversionPattern<ondrix::ir::QuantizeOp> {
public:
  using OpConversionPattern<ondrix::ir::QuantizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::QuantizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<ondrix::ondsp::ConvertOp>(
        op.getLoc(), op.getResult().getType(), adaptor.getInput(), op.getSrc(), op.getDst());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// The elementwise family lowers to one loop per operation whose body is the
// exact integer expression followed by the operation's single declared
// boundary. Widening to i32 first is what makes "exact" true of the body:
// nothing can lose a bit before the boundary the contract names.
template <typename SourceOp>
class ElementwiseOpLowering final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    RankedTensorType resultType = op.getResult().getType();
    int64_t extent = resultType.getDimSize(0);

    SmallVector<Value> sources;
    for (Value operand : adaptor.getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        sources.push_back(operand);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value extentValue = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), i16);
    auto loop = rewriter.create<scf::ForOp>(
        loc, zero, extentValue, one, ValueRange{empty},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange iterArgs) {
          SmallVector<Value> elements;
          for (Value source : sources)
            elements.push_back(builder.create<tensor::ExtractOp>(loc, source, position));
          Value value = emitBody(op, elements, context, i16, i32, loc, builder);
          Value inserted = builder.create<tensor::InsertOp>(loc, value, iterArgs.front(), position);
          builder.create<scf::YieldOp>(loc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
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
                        IntegerType i16, IntegerType i32, Location loc, OpBuilder &builder);
};

template <>
Value ElementwiseOpLowering<ondrix::ir::AddOp>::emitBody(ondrix::ir::AddOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType i16,
                                                         IntegerType i32, Location loc,
                                                         OpBuilder &builder) {
  // add_shift computes the sum one bit wider than its operands, so the i16
  // operands give the exact i17 sum and the scale is the only boundary.
  return builder.create<ondrix::ondsp::AddShiftOp>(
      loc, i16, elements[0], elements[1],
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::SubOp>::emitBody(ondrix::ir::SubOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType i16,
                                                         IntegerType i32, Location loc,
                                                         OpBuilder &builder) {
  return builder.create<ondrix::ondsp::SubShiftOp>(
      loc, i16, elements[0], elements[1],
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::MultOp>::emitBody(ondrix::ir::MultOp op,
                                                          ArrayRef<Value> elements,
                                                          MLIRContext *context, IntegerType i16,
                                                          IntegerType i32, Location loc,
                                                          OpBuilder &builder) {
  Value lhs = builder.create<arith::ExtSIOp>(loc, i32, elements[0]);
  Value rhs = builder.create<arith::ExtSIOp>(loc, i32, elements[1]);
  Value product = builder.create<arith::MulIOp>(loc, lhs, rhs);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, i16, product, narrowingScale(context, 15, op.getRounding(), op.getOverflow(), i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::AbsOp>::emitBody(ondrix::ir::AbsOp op,
                                                         ArrayRef<Value> elements,
                                                         MLIRContext *context, IntegerType i16,
                                                         IntegerType i32, Location loc,
                                                         OpBuilder &builder) {
  // Negating in i32 is what keeps |-32768| exact; the declared overflow then
  // decides the one input the destination cannot hold.
  Value wide = builder.create<arith::ExtSIOp>(loc, i32, elements[0]);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, i32);
  Value negated = builder.create<arith::SubIOp>(loc, zero, wide);
  Value magnitude = builder.create<arith::MaxSIOp>(loc, wide, negated);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, i16, magnitude,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::NegateOp>::emitBody(ondrix::ir::NegateOp op,
                                                            ArrayRef<Value> elements,
                                                            MLIRContext *context, IntegerType i16,
                                                            IntegerType i32, Location loc,
                                                            OpBuilder &builder) {
  Value wide = builder.create<arith::ExtSIOp>(loc, i32, elements[0]);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, i32);
  Value negated = builder.create<arith::SubIOp>(loc, zero, wide);
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, i16, negated,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::OffsetOp>::emitBody(ondrix::ir::OffsetOp op,
                                                            ArrayRef<Value> elements,
                                                            MLIRContext *context, IntegerType i16,
                                                            IntegerType i32, Location loc,
                                                            OpBuilder &builder) {
  Value bias = builder.create<arith::ConstantIntOp>(loc, op.getBiasAttr().getInt(), i16);
  return builder.create<ondrix::ondsp::AddShiftOp>(
      loc, i16, elements[0], bias,
      narrowingScale(context, 0, ondrix::ondsp::RoundingMode::TowardNegative, op.getOverflow(),
                     i16));
}

template <>
Value ElementwiseOpLowering<ondrix::ir::ShiftOp>::emitBody(ondrix::ir::ShiftOp op,
                                                           ArrayRef<Value> elements,
                                                           MLIRContext *context, IntegerType i16,
                                                           IntegerType i32, Location loc,
                                                           OpBuilder &builder) {
  int64_t amount = op.getAmountAttr().getInt();
  Value wide = builder.create<arith::ExtSIOp>(loc, i32, elements[0]);
  if (amount > 0) {
    // Exact in i32: 2^15 * 2^15 still fits, so only the narrowing can lose.
    Value shift = builder.create<arith::ConstantIntOp>(loc, amount, i32);
    wide = builder.create<arith::ShLIOp>(loc, wide, shift);
  }
  unsigned right = amount < 0 ? unsigned(-amount) : 0u;
  return builder.create<ondrix::ondsp::RoundShiftOp>(
      loc, i16, wide, narrowingScale(context, right, op.getRounding(), op.getOverflow(), i16));
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

class CxPhaseOpLowering final : public OpConversionPattern<ondrix::ir::CxPhaseOp> {
public:
  using OpConversionPattern<ondrix::ir::CxPhaseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::CxPhaseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    std::optional<SmallVector<int32_t>> table = buildArctangentTable();
    if (!table)
      return rewriter.notifyMatchFailure(op, "arctangent table entry is not tie-guard admissible");
    // The ratio division below writes nearest-even inline (its divisor is a
    // runtime value, so round_div cannot carry it). Same self-guard as exp2:
    // one operation must not follow two tie rules without a diagnostic.
    if (op.getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return rewriter.notifyMatchFailure(op, "cx_phase lowering implements nearest_even only");

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    IntegerType i16 = rewriter.getI16Type();
    IntegerType i32 = rewriter.getIntegerType(32);
    IntegerType i64 = rewriter.getIntegerType(64);
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
              loc, i32, builder.create<arith::TruncIOp>(loc, i16, packed));
          Value imaginary = builder.create<arith::ExtSIOp>(
              loc, i32,
              builder.create<arith::TruncIOp>(
                  loc, i16, builder.create<arith::ShRSIOp>(loc, packed, constant(16, i32))));
          Value zero32 = constant(0, i32);
          Value one32 = constant(1, i32);
          Value absReal = builder.create<arith::MaxSIOp>(
              loc, real, builder.create<arith::SubIOp>(loc, zero32, real));
          Value absImaginary = builder.create<arith::MaxSIOp>(
              loc, imaginary, builder.create<arith::SubIOp>(loc, zero32, imaginary));
          Value swapped =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, absImaginary, absReal);
          Value high = builder.create<arith::MaxSIOp>(loc, absReal, absImaginary);
          Value low = builder.create<arith::MinSIOp>(loc, absReal, absImaginary);
          // The origin has no argument; the divisor is forced to one so the
          // division is defined, and the declared value replaces the result.
          Value atOrigin =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, high, zero32);
          Value divisor = builder.create<arith::SelectOp>(loc, atOrigin, one32, high);

          // The ratio, rounded once. Both operands are non-negative, so the
          // truncating division is the floor and the remainder is the
          // Euclidean one; the tie test compares 2*remainder against the
          // divisor rather than forming a half that may not be an integer.
          Value numerator = builder.create<arith::ShLIOp>(
              loc, builder.create<arith::ExtSIOp>(loc, i64, low), constant(16, i64));
          Value wideDivisor = builder.create<arith::ExtSIOp>(loc, i64, divisor);
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
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, real, zero32);
          Value nonNegativeImaginary =
              builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, imaginary, zero32);
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

class Log2OpLowering final : public OpConversionPattern<ondrix::ir::Log2Op> {
public:
  using OpConversionPattern<ondrix::ir::Log2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ir::Log2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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
    std::optional<SmallVector<int32_t>> table = buildExp2Table();
    if (!table)
      return rewriter.notifyMatchFailure(op, "exp2 table entry is not tie-guard admissible");
    // The binade placement below writes nearest-even inline (its shift
    // amount is input-dependent, so round_shift cannot carry it). The
    // verifier pins the mode; this guard keeps the lowering from silently
    // applying two different rules if that pin is ever widened.
    if (op.getRounding() != ondrix::ondsp::RoundingMode::NearestEven)
      return rewriter.notifyMatchFailure(op, "exp2 lowering implements nearest_even only");

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

template <typename TrigOp, int64_t PhaseOffset>
class TrigOpLowering final : public OpConversionPattern<TrigOp> {
public:
  using OpConversionPattern<TrigOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TrigOp op, typename TrigOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getResult().getType();
    Value empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(),
                                                   resultType.getElementType());
    return lowerQ15Trig(op, adaptor.getInput(), empty, op.getNumeric(), PhaseOffset, rewriter);
  }
};

using SineOpLowering = TrigOpLowering<ondrix::ir::SineOp, 0>;
using CosineOpLowering = TrigOpLowering<ondrix::ir::CosineOp, 16384>;

} // namespace

void ondrix::conversion::populateOndrixElementwiseLoweringPatterns(RewritePatternSet &patterns) {
  patterns
      .add<QuantizeOpLowering, SineOpLowering, CosineOpLowering, Log2OpLowering, CxPhaseOpLowering,
           Exp2OpLowering, ElementwiseOpLowering<ondrix::ir::AddOp>,
           ElementwiseOpLowering<ondrix::ir::SubOp>, ElementwiseOpLowering<ondrix::ir::MultOp>,
           ElementwiseOpLowering<ondrix::ir::AbsOp>, ElementwiseOpLowering<ondrix::ir::NegateOp>,
           ElementwiseOpLowering<ondrix::ir::OffsetOp>, ElementwiseOpLowering<ondrix::ir::ShiftOp>>(
          patterns.getContext());
}
