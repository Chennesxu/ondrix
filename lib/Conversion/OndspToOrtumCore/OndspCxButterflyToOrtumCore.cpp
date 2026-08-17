#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPCXBUTTERFLYTOORTUMCORE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// The packed target rounding inventory; nearest_even deliberately maps to
// nothing so the default profile stays on the generic path.
static std::optional<ondrix::ortumcore::CxRounding>
selectTargetRounding(ondrix::ondsp::RoundingMode mode) {
  switch (mode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    return ondrix::ortumcore::CxRounding::TowardNegative;
  case ondrix::ondsp::RoundingMode::NearestTiesPositive:
    return ondrix::ortumcore::CxRounding::NearestTiesPositive;
  default:
    return std::nullopt;
  }
}

static ondrix::ortumcore::CxOverflow selectTargetOverflow(ondrix::ondsp::OverflowMode mode) {
  return mode == ondrix::ondsp::OverflowMode::Wrap ? ondrix::ortumcore::CxOverflow::Wrap
                                                   : ondrix::ortumcore::CxOverflow::Saturate;
}

struct TargetScale {
  ondrix::ortumcore::CxRounding rounding;
  ondrix::ortumcore::CxOverflow overflow;
  int64_t shift;
};

static std::optional<TargetScale> classifyTargetScale(ondrix::ondsp::ScaleAttr scale) {
  std::optional<ondrix::ortumcore::CxRounding> rounding = selectTargetRounding(scale.getRounding());
  if (!rounding || scale.getPreShiftLeft() != 0)
    return std::nullopt;
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  if (!destination || !destination.isSignless() || destination.getWidth() != 16)
    return std::nullopt;
  return TargetScale{*rounding, selectTargetOverflow(scale.getOverflow()),
                     int64_t(scale.getPostShiftRight())};
}

// The exact conjugate of a packed Q15 constant, or nullopt at the one
// unrepresentable point (imaginary component -32768).
static std::optional<uint32_t> conjugatePackedQ15(uint32_t bits) {
  int32_t imaginary = int16_t(bits >> 16);
  if (imaginary == -32768)
    return std::nullopt;
  return (uint32_t(uint16_t(-imaginary)) << 16) | (bits & 0xFFFF);
}

class ConvertOndspCxButterflyToOrtumCorePass final
    : public ondrix::impl::ConvertOndspCxButterflyToOrtumCoreBase<
          ConvertOndspCxButterflyToOrtumCorePass> {
public:
  void runOnOperation() override {
    conjugatedTables.clear();
    SmallVector<ondrix::ondsp::CxButterflyOp> candidates;
    getOperation()->walk([&](ondrix::ondsp::CxButterflyOp op) { candidates.push_back(op); });
    for (ondrix::ondsp::CxButterflyOp op : candidates)
      rewriteButterfly(op);
  }

private:
  // The twiddle operand as an exactly conjugated value: a scalar constant is
  // conjugated in place; a load from a compile-time table is redirected to a
  // once-materialized elementwise-conjugated table. Nullptr fails closed,
  // including at any table entry with imaginary -32768.
  Value materializeConjugatedTwiddle(ondrix::ondsp::CxButterflyOp op) {
    Location loc = op.getLoc();
    if (auto constant = op.getTwiddle().getDefiningOp<arith::ConstantOp>()) {
      auto twiddle = dyn_cast<IntegerAttr>(constant.getValue());
      if (!twiddle)
        return Value();
      std::optional<uint32_t> conjugated =
          conjugatePackedQ15(uint32_t(twiddle.getValue().getZExtValue()));
      if (!conjugated)
        return Value();
      OpBuilder builder(op);
      return builder.create<arith::ConstantOp>(loc,
                                               builder.getI32IntegerAttr(int32_t(*conjugated)));
    }
    auto extract = op.getTwiddle().getDefiningOp<tensor::ExtractOp>();
    auto table =
        extract ? extract.getTensor().getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
    auto elements = table ? dyn_cast<DenseIntElementsAttr>(table.getValue()) : nullptr;
    if (!elements || !elements.getElementType().isSignlessInteger(32))
      return Value();
    Value conjugatedTable = conjugatedTables.lookup(table);
    if (!conjugatedTable) {
      SmallVector<int32_t> values;
      values.reserve(elements.getNumElements());
      for (APInt element : elements) {
        std::optional<uint32_t> conjugated = conjugatePackedQ15(uint32_t(element.getZExtValue()));
        if (!conjugated)
          return Value();
        values.push_back(int32_t(*conjugated));
      }
      OpBuilder builder(table);
      conjugatedTable = builder.create<arith::ConstantOp>(
          table.getLoc(), DenseElementsAttr::get(cast<ShapedType>(elements.getType()),
                                                 llvm::ArrayRef<int32_t>(values)));
      conjugatedTables[table] = conjugatedTable;
    }
    OpBuilder builder(op);
    return builder.create<tensor::ExtractOp>(loc, conjugatedTable, extract.getIndices());
  }

  void rewriteButterfly(ondrix::ondsp::CxButterflyOp op) {
    if (!op.getA().getType().isSignlessInteger(32) ||
        op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo)
      return;
    ondrix::ondsp::CxButterflyVariant variant =
        op.getVariant().value_or(ondrix::ondsp::CxButterflyVariant::Plain);
    bool cross = variant == ondrix::ondsp::CxButterflyVariant::Cross ||
                 variant == ondrix::ondsp::CxButterflyVariant::UnitCross;
    bool unit = variant == ondrix::ondsp::CxButterflyVariant::Unit ||
                variant == ondrix::ondsp::CxButterflyVariant::UnitCross;
    std::optional<TargetScale> output = classifyTargetScale(op.getOutputScale());
    if (!output || output->shift > 1)
      return;

    // The unit variants have no product stage: b feeds the packed pair
    // combine directly and exactly. The cross form consumes its operand in
    // the swapped packing, so the exact unit cross swaps b's halves first.
    Value rotated = op.getB();
    OpBuilder builder(op);
    Location loc = op.getLoc();
    if (unit && cross) {
      Value sixteen = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(16));
      Value high = builder.create<arith::ShRUIOp>(loc, rotated, sixteen);
      Value low = builder.create<arith::ShLIOp>(loc, rotated, sixteen);
      rotated = builder.create<arith::OrIOp>(loc, low, high);
    }
    if (!unit) {
      std::optional<TargetScale> product = classifyTargetScale(op.getProductScale());
      if (!product || product->shift != 15)
        return;
      Value conjugate = materializeConjugatedTwiddle(op);
      if (!conjugate)
        return;
      // The cross combine consumes the pair's shared product through the
      // swapped (real-high) packing, which is exactly what makes a -+ j*t
      // free on the target.
      rotated = builder.create<ondrix::ortumcore::CxMulConjOp>(
          loc, builder.getI32Type(), op.getB(), conjugate, uint64_t(product->shift),
          product->rounding, product->overflow,
          cross ? ondrix::ortumcore::CxLayout::RealHi : ondrix::ortumcore::CxLayout::ImagHi);
    }
    auto pair = builder.create<ondrix::ortumcore::CxBflyOp>(
        loc, builder.getI32Type(), builder.getI32Type(), op.getA(), rotated,
        uint64_t(output->shift), output->rounding, output->overflow,
        cross ? ondrix::ortumcore::CxBflyVariant::Cross : ondrix::ortumcore::CxBflyVariant::Plain);
    op.getOut0().replaceAllUsesWith(pair.getOut0());
    op.getOut1().replaceAllUsesWith(pair.getOut1());
    op.erase();
  }

  llvm::DenseMap<Operation *, Value> conjugatedTables;
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspCxButterflyToOrtumCorePass() {
  return std::make_unique<ConvertOndspCxButterflyToOrtumCorePass>();
}
