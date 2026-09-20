#include "OrtumCoreLoweringSupport.h"

#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace ondrix::conversion {

ortumcore::AccumulatorDomain getOrtumCoreAccumulatorDomain(ondsp::AccType accumulator) {
  return {cast<IntegerType>(accumulator.getStorage()).getWidth(), accumulator.getFrac(),
          accumulator.getSignedness(), accumulator.getUpdateOverflow()};
}

bool isOrtumCoreLaneDomain(ondsp::AccType accumulator) {
  return ortumcore::OrtumCoreTargetProfile().supportsAccumulator(
      getOrtumCoreAccumulatorDomain(accumulator));
}

bool isOrtumCoreMacPolicy(ondsp::MacOp op) {
  // The pre-filter keeps the shared semantic inference silent: it only emits
  // diagnostics on shapes the pre-filter already rejects.
  bool q15Full = ondsp::isSignedQ15(op.getNumeric()) &&
                 op.getProduct().getSelection() == ondsp::ProductSelection::Full;
  bool q31RawHigh = ondsp::isSignedQ31(op.getNumeric()) && ondsp::isRawHighProduct(op.getProduct());
  if (!q15Full && !q31RawHigh)
    return false;
  FailureOr<ondsp::ProductSemantics> semantics =
      ondsp::inferProductSemantics(op, op.getNumeric(), op.getProduct());
  if (failed(semantics))
    return false;
  ondsp::FixedAttr numeric = op.getNumeric();
  ortumcore::ProductDomain product{cast<IntegerType>(numeric.getStorage()).getWidth(),
                                   numeric.getFrac(), numeric.getSignedness(), *semantics};
  return ortumcore::OrtumCoreTargetProfile().supportsMac(
      product, getOrtumCoreAccumulatorDomain(op.getAcc().getType()));
}

std::optional<OrtumCoreExportPolicy> classifyOrtumCoreExport(ondsp::AccExportOp op) {
  ondsp::AccType accumulator = op.getAcc().getType();
  auto storage = dyn_cast<IntegerType>(op.getDst().getStorage());
  int64_t shift = int64_t(accumulator.getFrac()) - int64_t(op.getDst().getFrac());
  if (!storage || (storage.getWidth() != 32 && storage.getWidth() != 16) ||
      !ortumcore::OrtumCoreTargetProfile().supportsExport(
          getOrtumCoreAccumulatorDomain(accumulator), op.getRounding(), op.getOverflow(), shift))
    return std::nullopt;
  return OrtumCoreExportPolicy{shift, storage, op.getRounding()};
}

Value emitOrtumCoreReadout(OpBuilder &builder, Location loc, Value acc,
                           const OrtumCoreExportPolicy &policy) {
  constexpr int64_t kMaxCapabilityShift = 15;
  int64_t tail = std::max<int64_t>(0, policy.shift - kMaxCapabilityShift);
  Value out;
  if (policy.rounding == ondsp::RoundingMode::NearestTiesPositive && policy.shift > 0 &&
      tail == 0) {
    // floor((acc + 2^(s-1)) / 2^s) == (r >> 1) + (r & 1) for r = acc >> (s-1),
    // total in i32 (no widening), and in the admitted range the narrower
    // readout cannot clip (Passes.td carries the argument), so it is exact.
    Value narrower =
        builder.create<ortumcore::AccOutOp>(loc, builder.getI32Type(), acc, policy.shift - 1);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value halved = builder.create<arith::ShRSIOp>(loc, narrower, one);
    Value roundBit = builder.create<arith::AndIOp>(loc, narrower, one);
    out = builder.create<arith::AddIOp>(loc, halved, roundBit);
  } else if (policy.rounding == ondsp::RoundingMode::NearestTiesPositive && tail > 0) {
    // Past the capability range the half-add commutes over the always-exact
    // max-shift readout (2^15 divides 2^(s-1)), landing as one base add on
    // the 25-bit value before the exact nested tail shift.
    out = builder.create<ortumcore::AccOutOp>(loc, builder.getI32Type(), acc, kMaxCapabilityShift);
    Value half = builder.create<arith::ConstantIntOp>(loc, int64_t(1) << (tail - 1), 32);
    Value sum = builder.create<arith::AddIOp>(loc, out, half);
    Value amount = builder.create<arith::ConstantIntOp>(loc, tail, 32);
    out = builder.create<arith::ShRSIOp>(loc, sum, amount);
  } else {
    out = builder.create<ortumcore::AccOutOp>(loc, builder.getI32Type(), acc, policy.shift - tail);
    if (tail > 0) {
      // floor nests exactly over the base tail shift.
      Value amount = builder.create<arith::ConstantIntOp>(loc, tail, 32);
      out = builder.create<arith::ShRSIOp>(loc, out, amount);
    }
  }
  // The i16 destination clamp composes exactly: the readout's wider i32
  // saturation cannot change a subsequent narrower clamp (Passes.td carries
  // the argument).
  return emitSignedSaturatingNarrow(builder, loc, out, policy.storage);
}

std::optional<ortumcore::CxRounding> selectPackedComplexRounding(ondsp::RoundingMode mode) {
  switch (mode) {
  case ondsp::RoundingMode::TowardNegative:
    return ortumcore::CxRounding::TowardNegative;
  case ondsp::RoundingMode::NearestTiesPositive:
    return ortumcore::CxRounding::NearestTiesPositive;
  case ondsp::RoundingMode::NearestEven:
  case ondsp::RoundingMode::TowardZero:
    return std::nullopt;
  }
  return std::nullopt;
}

Value emitSignedSaturatingNarrow(OpBuilder &builder, Location loc, Value value,
                                 IntegerType storage) {
  if (storage.getWidth() == 32)
    return value;
  int64_t limit = int64_t{1} << (storage.getWidth() - 1);
  Value minimum = builder.create<arith::ConstantIntOp>(loc, -limit, 32);
  Value maximum = builder.create<arith::ConstantIntOp>(loc, limit - 1, 32);
  Value belowMinimum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, value, minimum);
  Value lowerClamped = builder.create<arith::SelectOp>(loc, belowMinimum, minimum, value);
  Value aboveMaximum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lowerClamped, maximum);
  Value clamped = builder.create<arith::SelectOp>(loc, aboveMaximum, maximum, lowerClamped);
  return builder.create<arith::TruncIOp>(loc, storage, clamped);
}

} // namespace ondrix::conversion
