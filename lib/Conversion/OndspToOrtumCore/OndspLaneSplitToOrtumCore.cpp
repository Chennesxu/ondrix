#include "OrtumCoreLoweringSupport.h"
#include "ondrix/Analysis/ConstantSequenceAnalysis.h"
#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"
#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <limits>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPLANESPLITTOORTUMCORE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

using ondrix::analysis::FixedPointRawInterval;

constexpr int64_t kMaxTableEntries = 4096;
// Two dual steps at least: one pair saves nothing against the merge.
constexpr int64_t kMinTerms = 4;
constexpr unsigned kLaneWordBits = 32;

/// A constant multiplicand the certificate may read: a literal, or one element
/// of a constant global at a constant index.
std::optional<APInt> getConstantTerm(Value value) {
  APInt literal;
  if (matchPattern(value, m_ConstantInt(&literal)))
    return literal;
  auto load = value.getDefiningOp<memref::LoadOp>();
  if (!load || load.getIndices().size() != 1)
    return std::nullopt;
  std::optional<int64_t> index = getConstantIntValue(load.getIndices().front());
  FailureOr<ondrix::ConstantIntegerMemRefFacts> table =
      ondrix::analyzeConstantIntegerMemRef(load.getMemRef(), kMaxTableEntries);
  if (!index || failed(table) || *index < 0 || *index >= table->getSequence().getElementCount())
    return std::nullopt;
  return table->getSequence().getValues()[*index];
}

/// A runtime coefficient read at a constant index through a declared l1 bound.
struct BoundedRead {
  Value table;
  int64_t bound;
  int64_t index;
};

std::optional<BoundedRead> getBoundedRead(Value value) {
  auto load = value.getDefiningOp<memref::LoadOp>();
  if (!load || load.getIndices().size() != 1)
    return std::nullopt;
  auto declared = load.getMemRef().getDefiningOp<ondrix::ondsp::AssumeL1BoundOp>();
  std::optional<int64_t> index = getConstantIntValue(load.getIndices().front());
  if (!declared || !index)
    return std::nullopt;
  return BoundedRead{load.getMemRef(), declared.getBound(), *index};
}

/// The runtime stream's element index when `value` reads it at a constant one.
std::optional<int64_t> getStreamIndex(Value value, Value stream) {
  auto load = value.getDefiningOp<memref::LoadOp>();
  if (!load || load.getMemRef() != stream || load.getIndices().size() != 1)
    return std::nullopt;
  return getConstantIntValue(load.getIndices().front());
}

struct LaneSplit {
  ondrix::ondsp::AccZeroOp seed;
  SmallVector<ondrix::ondsp::MacOp> chain;
  ondrix::ondsp::AccExportOp exported;
  ondrix::conversion::OrtumCoreExportPolicy policy;
  SmallVector<Value> signals;
  SmallVector<Value> coefficients;
  Value stream;
  // Set when the coefficients are a runtime table under a declared bound; it
  // is then word-read too, so it shares the stream's alignment test.
  Value table;
  bool sumFitsWord = false;
};

/// A zero-seeded single-lane Q15 chain whose even and odd terms each provably
/// stay inside one signed word, read from one runtime stream in pair order.
std::optional<LaneSplit> matchLaneSplit(ondrix::ondsp::AccZeroOp seed) {
  auto accumulator = cast<ondrix::ondsp::AccType>(seed.getType());
  if (accumulator.getLanes() != 1 || !ondrix::conversion::isOrtumCoreLaneDomain(accumulator))
    return std::nullopt;

  LaneSplit split;
  split.seed = seed;
  Value acc = seed.getResult();
  while (true) {
    if (!acc.hasOneUse())
      return std::nullopt;
    Operation *user = *acc.getUsers().begin();
    if (user->getBlock() != seed->getBlock())
      return std::nullopt;
    if (auto mac = dyn_cast<ondrix::ondsp::MacOp>(user)) {
      if (mac.getAcc() != acc || !ondrix::ondsp::isSignedQ15(mac.getNumeric()) ||
          !ondrix::conversion::isOrtumCoreMacPolicy(mac) || !mac.getLhs().getType().isInteger(16))
        return std::nullopt;
      split.chain.push_back(mac);
      acc = mac.getResult();
      continue;
    }
    auto exported = dyn_cast<ondrix::ondsp::AccExportOp>(user);
    if (!exported)
      return std::nullopt;
    std::optional<ondrix::conversion::OrtumCoreExportPolicy> policy =
        ondrix::conversion::classifyOrtumCoreExport(exported);
    if (!policy || exported.getResult().getType() != policy->storage)
      return std::nullopt;
    split.exported = exported;
    split.policy = *policy;
    break;
  }
  if (static_cast<int64_t>(split.chain.size()) < kMinTerms)
    return std::nullopt;

  ondrix::ondsp::FixedAttr numeric = split.chain.front().getNumeric();
  std::optional<FixedPointRawInterval> lanes[2];
  std::optional<BoundedRead> firstBounded;
  bool anyConstant = false;
  for (auto [position, mac] : llvm::enumerate(split.chain)) {
    Value signal = mac.getLhs();
    Value term = mac.getRhs();
    std::optional<APInt> coefficient = getConstantTerm(term);
    std::optional<BoundedRead> bounded = coefficient ? std::nullopt : getBoundedRead(term);
    if (!coefficient && !bounded) {
      std::swap(signal, term);
      coefficient = getConstantTerm(term);
      bounded = coefficient ? std::nullopt : getBoundedRead(term);
    }
    // One certificate per chain: every term constant, or every term read in
    // order from one table whose declared bound covers the whole sum.
    if (bounded) {
      if (anyConstant || !mac.getLhs().getType().isInteger(16) ||
          (firstBounded && (firstBounded->table != bounded->table ||
                            bounded->index != firstBounded->index + int64_t(position))) ||
          (!firstBounded && bounded->index % 2 != 0))
        return std::nullopt;
      if (!firstBounded)
        firstBounded = bounded;
    } else if (firstBounded) {
      return std::nullopt;
    }
    anyConstant |= coefficient.has_value();
    if (!bounded && (!coefficient || coefficient->getBitWidth() != 16))
      return std::nullopt;
    if (!split.stream)
      if (auto load = signal.getDefiningOp<memref::LoadOp>())
        split.stream = load.getMemRef();
    std::optional<int64_t> index = getStreamIndex(signal, split.stream);
    std::optional<int64_t> first =
        split.signals.empty() ? index : getStreamIndex(split.signals.front(), split.stream);
    // The pair (2k, 2k + 1) must be one aligned word of the stream, or every
    // dual step packs its halves and the split buys nothing.
    if (!index || !first || *first % 2 != 0 || *index != *first + int64_t(position))
      return std::nullopt;
    split.signals.push_back(signal);
    split.coefficients.push_back(term);
    if (bounded)
      continue;
    FailureOr<FixedPointRawInterval> product =
        ondrix::analysis::computeSignedFullProductInterval(numeric, *coefficient);
    if (failed(product))
      return std::nullopt;
    std::optional<FixedPointRawInterval> &lane = lanes[position % 2];
    if (!lane) {
      lane = *product;
    } else {
      FailureOr<FixedPointRawInterval> sum =
          ondrix::analysis::addFixedPointRawIntervals(*lane, *product);
      if (failed(sum))
        return std::nullopt;
      lane = *sum;
    }
  }

  auto stream = dyn_cast<MemRefType>(split.stream.getType());
  if (!stream || stream.getRank() != 1 || !stream.hasStaticShape() ||
      !stream.getLayout().isIdentity() || !stream.getElementType().isInteger(16))
    return std::nullopt;
  // Under a declared bound B every partial sum of either lane, and of the
  // whole chain, is at most 2^15 * (B - 1) in magnitude, the terms being
  // distinct elements of the bounded table.
  if (firstBounded) {
    split.table = firstBounded->table;
    int64_t magnitude = (int64_t{1} << 15) * (firstBounded->bound - 1);
    split.sumFitsWord = magnitude <= std::numeric_limits<int32_t>::max();
    if (!split.sumFitsWord)
      return std::nullopt;
    return split;
  }
  // Every partial sum of a lane lies inside the lane's interval, since each
  // product interval contains zero, so a lane that fits one word at its end
  // never reaches the accumulator rail and reads out exactly at shift 0.
  if (!ondrix::analysis::fitsSignedImplementationWidth(*lanes[0], kLaneWordBits) ||
      !ondrix::analysis::fitsSignedImplementationWidth(*lanes[1], kLaneWordBits))
    return std::nullopt;
  FailureOr<FixedPointRawInterval> sum =
      ondrix::analysis::addFixedPointRawIntervals(*lanes[0], *lanes[1]);
  if (failed(sum))
    return std::nullopt;
  split.sumFitsWord = ondrix::analysis::fitsSignedImplementationWidth(*sum, kLaneWordBits);
  if (ondrix::conversion::getOrtumCoreReadoutShift(split.policy) == 0 && !split.sumFitsWord)
    return std::nullopt;
  return split;
}

/// The operand as the branch reads it: a load is re-issued there, so each
/// branch pays only for its own reads.
Value materialize(OpBuilder &builder, Value value) {
  if (auto load = value.getDefiningOp<memref::LoadOp>())
    return builder.clone(*load)->getResult(0);
  return value;
}

Value emitSplitChain(OpBuilder &builder, Location loc, LaneSplit &split) {
  Type laneType = ondrix::ortumcore::AccumType::get(builder.getContext());
  Value lane0 = builder.create<ondrix::ortumcore::AccInitOp>(loc, laneType);
  Value lane1 = builder.create<ondrix::ortumcore::AccInitOp>(loc, laneType);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 16);
  for (size_t position = 0; position < split.signals.size(); position += 2) {
    bool paired = position + 1 < split.signals.size();
    // An odd last term rides lane 0 against an exact zero product.
    Value value0 = materialize(builder, split.signals[position]);
    Value coefficient0 = materialize(builder, split.coefficients[position]);
    Value value1 = paired ? materialize(builder, split.signals[position + 1]) : zero;
    Value coefficient1 = paired ? materialize(builder, split.coefficients[position + 1]) : zero;
    auto step = builder.create<ondrix::ortumcore::DmacOp>(
        loc, laneType, laneType, lane0, lane1, value0, coefficient0, value1, coefficient1);
    lane0 = step.getOut0();
    lane1 = step.getOut1();
  }
  Type word = builder.getI32Type();
  Value sum0 = builder.create<ondrix::ortumcore::AccOutOp>(loc, word, lane0, 0);
  Value sum1 = builder.create<ondrix::ortumcore::AccOutOp>(loc, word, lane1, 0);
  auto readout = [&](int64_t shift) -> Value {
    if (split.sumFitsWord) {
      Value sum = builder.create<arith::AddIOp>(loc, sum0, sum1);
      if (shift == 0)
        return sum;
      Value amount = builder.create<arith::ConstantIntOp>(loc, shift, 32);
      return builder.create<arith::ShRSIOp>(loc, sum, amount);
    }
    // floor((a + b) / 2) == (a >> 1) + (b >> 1) + (a & b & 1) in one word,
    // and the shifted sum of two word-sized lanes fits a word, so the
    // capability's saturation never fires on it.
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value half0 = builder.create<arith::ShRSIOp>(loc, sum0, one);
    Value half1 = builder.create<arith::ShRSIOp>(loc, sum1, one);
    Value both = builder.create<arith::AndIOp>(loc, sum0, sum1);
    Value carry = builder.create<arith::AndIOp>(loc, both, one);
    Value half =
        builder.create<arith::AddIOp>(loc, builder.create<arith::AddIOp>(loc, half0, half1), carry);
    if (shift == 1)
      return half;
    Value rest = builder.create<arith::ConstantIntOp>(loc, shift - 1, 32);
    return builder.create<arith::ShRSIOp>(loc, half, rest);
  };
  return ondrix::conversion::emitOrtumCoreReadout(builder, loc, readout, split.policy);
}

Value emitOrderedChain(OpBuilder &builder, LaneSplit &split) {
  Value acc = builder.clone(*split.seed)->getResult(0);
  for (auto [position, mac] : llvm::enumerate(split.chain)) {
    IRMapping mapping;
    mapping.map(mac.getAcc(), acc);
    mapping.map(split.signals[position], materialize(builder, split.signals[position]));
    mapping.map(split.coefficients[position], materialize(builder, split.coefficients[position]));
    acc = builder.clone(*mac, mapping)->getResult(0);
  }
  IRMapping mapping;
  mapping.map(split.exported.getAcc(), acc);
  return builder.clone(*split.exported, mapping)->getResult(0);
}

void rewriteLaneSplit(LaneSplit &split) {
  Location loc = split.exported.getLoc();
  OpBuilder builder(split.exported);
  Value pointer = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, split.stream);
  if (split.table)
    pointer = builder.create<arith::OrIOp>(
        loc, pointer, builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, split.table));
  Value mask = builder.create<arith::ConstantIndexOp>(loc, 3);
  Value misalignment = builder.create<arith::AndIOp>(loc, pointer, mask);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value aligned = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, misalignment, zero);
  auto branch = builder.create<scf::IfOp>(
      loc, aligned,
      [&](OpBuilder &then, Location thenLoc) {
        then.create<memref::AssumeAlignmentOp>(thenLoc, split.stream, 4);
        if (split.table)
          then.create<memref::AssumeAlignmentOp>(thenLoc, split.table, 4);
        then.create<scf::YieldOp>(thenLoc, emitSplitChain(then, thenLoc, split));
      },
      [&](OpBuilder &otherwise, Location elseLoc) {
        otherwise.create<scf::YieldOp>(elseLoc, emitOrderedChain(otherwise, split));
      });

  split.exported.getResult().replaceAllUsesWith(branch.getResult(0));
  SmallVector<Operation *> operands;
  for (auto [signal, coefficient] : llvm::zip(split.signals, split.coefficients))
    for (Value value : {signal, coefficient})
      if (Operation *load = value.getDefiningOp<memref::LoadOp>())
        operands.push_back(load);
  split.exported.erase();
  for (ondrix::ondsp::MacOp mac : llvm::reverse(split.chain))
    mac.erase();
  split.seed.erase();
  for (Operation *load : operands)
    if (load->use_empty())
      load->erase();
}

class ConvertOndspLaneSplitToOrtumCorePass final
    : public ondrix::impl::ConvertOndspLaneSplitToOrtumCoreBase<
          ConvertOndspLaneSplitToOrtumCorePass> {
public:
  using ondrix::impl::ConvertOndspLaneSplitToOrtumCoreBase<
      ConvertOndspLaneSplitToOrtumCorePass>::ConvertOndspLaneSplitToOrtumCoreBase;

  void runOnOperation() override {
    SmallVector<LaneSplit> splits;
    getOperation().walk([&](ondrix::ondsp::AccZeroOp seed) {
      if (std::optional<LaneSplit> split = matchLaneSplit(seed))
        splits.push_back(std::move(*split));
    });
    for (LaneSplit &split : splits)
      rewriteLaneSplit(split);
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspLaneSplitToOrtumCorePass() {
  return std::make_unique<ConvertOndspLaneSplitToOrtumCorePass>();
}
