#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Target/OrtumCore/OrtumCoreCapabilities.h"

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

// The packed target rounding inventory. nearest_even and toward_zero
// deliberately map to nothing so those profiles stay on the generic path,
// and a newly declared mode lands there too (plus a -Wswitch finding here)
// instead of borrowing an inventory member.
static std::optional<ondrix::ortumcore::CxRounding>
selectTargetRounding(ondrix::ondsp::RoundingMode mode) {
  switch (mode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    return ondrix::ortumcore::CxRounding::TowardNegative;
  case ondrix::ondsp::RoundingMode::NearestTiesPositive:
    return ondrix::ortumcore::CxRounding::NearestTiesPositive;
  case ondrix::ondsp::RoundingMode::NearestEven:
  case ondrix::ondsp::RoundingMode::TowardZero:
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<ondrix::ortumcore::CxOverflow>
selectTargetOverflow(ondrix::ondsp::OverflowMode mode) {
  switch (mode) {
  case ondrix::ondsp::OverflowMode::Wrap:
    return ondrix::ortumcore::CxOverflow::Wrap;
  case ondrix::ondsp::OverflowMode::Saturate:
    return ondrix::ortumcore::CxOverflow::Saturate;
  }
  return std::nullopt;
}

struct TargetScale {
  ondrix::ortumcore::CxRounding rounding;
  ondrix::ortumcore::CxOverflow overflow;
  int64_t shift;
};

static std::optional<TargetScale> classifyTargetScale(ondrix::ondsp::ScaleAttr scale) {
  std::optional<ondrix::ortumcore::CxRounding> rounding = selectTargetRounding(scale.getRounding());
  std::optional<ondrix::ortumcore::CxOverflow> overflow = selectTargetOverflow(scale.getOverflow());
  if (!rounding || !overflow || scale.getPreShiftLeft() != 0)
    return std::nullopt;
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  if (!destination || !destination.isSignless() || destination.getWidth() != 16)
    return std::nullopt;
  return TargetScale{*rounding, *overflow, int64_t(scale.getPostShiftRight())};
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
    SmallVector<ondrix::ondsp::BitrevAddOp> walks;
    SmallVector<Operation *> shifts;
    getOperation()->walk([&](Operation *op) {
      if (auto butterfly = dyn_cast<ondrix::ondsp::CxButterflyOp>(op))
        candidates.push_back(butterfly);
      else if (auto walk = dyn_cast<ondrix::ondsp::BitrevAddOp>(op))
        walks.push_back(walk);
      else if (isa<ondrix::ondsp::AddShiftOp, ondrix::ondsp::SubShiftOp>(op))
        shifts.push_back(op);
    });
    for (ondrix::ondsp::CxButterflyOp op : candidates)
      rewriteButterfly(op);
    for (ondrix::ondsp::BitrevAddOp op : walks)
      rewriteBitrevAdd(op);
    for (Operation *op : shifts)
      rewriteScaledShift(op);
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

  // Top-aligned rev32 composition: shifting the low `width` bits to the top
  // makes the 32-bit reversed-carry add compute the width-bit one exactly -
  // the sum's carry out of the window lands below the extracted bits.
  void rewriteBitrevAdd(ondrix::ondsp::BitrevAddOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    IntegerType i32 = builder.getI32Type();
    Value amount = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(static_cast<int32_t>(32 - op.getWidth())));
    auto scaled = [&](Value value) {
      Value narrowed = builder.create<arith::IndexCastUIOp>(loc, i32, value);
      return builder.create<arith::ShLIOp>(loc, narrowed, amount).getResult();
    };
    Value walked = builder.create<ondrix::ortumcore::BitrevAddOp>(loc, i32, scaled(op.getBase()),
                                                                  scaled(op.getStep()));
    Value extracted = builder.create<arith::ShRUIOp>(loc, walked, amount);
    Value result = builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(), extracted);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }

  // A standalone add/sub_shift whose scale IS the target's scaled saturating
  // add/sub (the exact attribute shape the OrtumCore emulation regenerates)
  // selects directly; every other scale stays generic.
  void rewriteScaledShift(Operation *op) {
    auto scale = cast<ondrix::ondsp::ScaleAttr>(op->getAttr("scale"));
    auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
    if (scale.getPreShiftLeft() != 0 ||
        uint64_t(scale.getPostShiftRight()) >
            ondrix::ortumcore::getShiftedSaturatingI32ScaledBinaryDomain().maxShift ||
        scale.getRounding() != ondrix::ondsp::RoundingMode::TowardNegative ||
        scale.getOverflow() != ondrix::ondsp::OverflowMode::Saturate || !destination ||
        !destination.isSignless() || destination.getWidth() != 32 ||
        !op->getResult(0).getType().isSignlessInteger(32) ||
        !op->getOperand(0).getType().isSignlessInteger(32) ||
        !op->getOperand(1).getType().isSignlessInteger(32))
      return;
    OpBuilder builder(op);
    Location loc = op->getLoc();
    IntegerType i32 = builder.getI32Type();
    uint64_t shift = uint64_t(scale.getPostShiftRight());
    Value selected = isa<ondrix::ondsp::SubShiftOp>(op)
                         ? builder
                               .create<ondrix::ortumcore::SatShiftSubOp>(
                                   loc, i32, op->getOperand(0), op->getOperand(1), shift)
                               .getResult()
                         : builder
                               .create<ondrix::ortumcore::SatShiftAddOp>(
                                   loc, i32, op->getOperand(0), op->getOperand(1), shift)
                               .getResult();
    op->getResult(0).replaceAllUsesWith(selected);
    op->erase();
  }

  // Selection by decomposition onto the existing scalar capabilities; the
  // equivalence argument is the pass description in Passes.td.
  bool rewriteRawHighQ31Butterfly(ondrix::ondsp::CxButterflyOp op) {
    if (!op.getA().getType().isSignlessInteger(64) ||
        op.getLayout().getLayout() != ondrix::ondsp::ComplexLayout::PackedI32ImagHiRealLo ||
        op.getProduct().getSelection() != ondrix::ondsp::ProductSelection::HighRaw ||
        op.getVariant().value_or(ondrix::ondsp::CxButterflyVariant::Plain) !=
            ondrix::ondsp::CxButterflyVariant::Plain)
      return false;
    // The raw-high profile's boundaries are fixed by the operation's own
    // verifier - product scale (left one, floor, saturating) and an output
    // shift of 0 or 1 - so only the shift is read here.
    int64_t outputShift = op.getOutputScale().getPostShiftRight();
    if (uint64_t(outputShift) >
        ondrix::ortumcore::getShiftedSaturatingI32ScaledBinaryDomain().maxShift)
      return false;

    OpBuilder builder(op);
    Location loc = op.getLoc();
    IntegerType i32 = builder.getI32Type();
    IntegerType i64 = builder.getI64Type();
    Type accumulator = ondrix::ortumcore::AccumType::get(builder.getContext());
    Value halfWidth = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(32));
    auto real = [&](Value packed) {
      return builder.create<arith::TruncIOp>(loc, i32, packed).getResult();
    };
    auto imaginary = [&](Value packed) {
      Value high = builder.create<arith::ShRUIOp>(loc, packed, halfWidth);
      return builder.create<arith::TruncIOp>(loc, i32, high).getResult();
    };
    Value ar = real(op.getA());
    Value ai = imaginary(op.getA());
    Value br = real(op.getB());
    Value bi = imaginary(op.getB());
    Value wr = real(op.getTwiddle());
    Value wi = imaginary(op.getTwiddle());

    // One cross-term pair per accumulator web. Each raw-high term is bounded by
    // 2^30, so the pair stays far inside the 40-bit range and the saturating
    // updates are exact. Reading out at shift 0 and doubling through the
    // saturating add realizes the product scale's left shift exactly: clamping
    // before a monotone doubling that clamps again agrees with clamping after.
    auto crossTerm = [&](Value lhs, Value rhs, Value subLhs, Value subRhs, bool subtract) {
      Value web = builder.create<ondrix::ortumcore::AccInitOp>(loc, accumulator);
      web = builder.create<ondrix::ortumcore::Q31MacAddOp>(loc, accumulator, web, lhs, rhs);
      web = subtract
                ? builder
                      .create<ondrix::ortumcore::Q31MacSubOp>(loc, accumulator, web, subLhs, subRhs)
                      .getResult()
                : builder
                      .create<ondrix::ortumcore::Q31MacAddOp>(loc, accumulator, web, subLhs, subRhs)
                      .getResult();
      Value read = builder.create<ondrix::ortumcore::AccOutOp>(loc, i32, web, uint64_t(0));
      return builder.create<ondrix::ortumcore::SatShiftAddOp>(loc, i32, read, read, uint64_t(0))
          .getResult();
    };
    Value tr = crossTerm(br, wr, bi, wi, /*subtract=*/true);
    Value ti = crossTerm(br, wi, bi, wr, /*subtract=*/false);

    auto pack = [&](Value component, Value high) {
      Value low = builder.create<arith::ExtUIOp>(loc, i64, component);
      Value shifted = builder.create<arith::ShLIOp>(
          loc, builder.create<arith::ExtUIOp>(loc, i64, high), halfWidth);
      return builder.create<arith::OrIOp>(loc, shifted, low).getResult();
    };
    auto stage = [&](Value component, Value term, bool subtract) {
      if (subtract)
        return builder
            .create<ondrix::ortumcore::SatShiftSubOp>(loc, i32, component, term,
                                                      uint64_t(outputShift))
            .getResult();
      return builder
          .create<ondrix::ortumcore::SatShiftAddOp>(loc, i32, component, term,
                                                    uint64_t(outputShift))
          .getResult();
    };
    // Sequenced deliberately: two calls inside one argument list would emit in
    // an order the language does not fix.
    Value sumReal = stage(ar, tr, /*subtract=*/false);
    Value sumImaginary = stage(ai, ti, /*subtract=*/false);
    Value differenceReal = stage(ar, tr, /*subtract=*/true);
    Value differenceImaginary = stage(ai, ti, /*subtract=*/true);
    op.getOut0().replaceAllUsesWith(pack(sumReal, sumImaginary));
    op.getOut1().replaceAllUsesWith(pack(differenceReal, differenceImaginary));
    op.erase();
    return true;
  }

  void rewriteButterfly(ondrix::ondsp::CxButterflyOp op) {
    if (rewriteRawHighQ31Butterfly(op))
      return;
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
