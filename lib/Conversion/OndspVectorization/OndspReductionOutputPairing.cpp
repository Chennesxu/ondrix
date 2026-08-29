#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace ondrix {
#define GEN_PASS_DEF_PAIRONDSPFIXEDREDUCTIONOUTPUTS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Reduction length above which the per-index chain stays an `scf.for`. It is
/// the batcher's measured unroll bound in `ScalarizeOndspFixedReduceMac`.
constexpr int64_t kMaxUnrolledLength = 64;

/// Largest total index count a straight-line pair body may emit. Above it the
/// column-loop rewrite keeps a loop over the output pairs.
constexpr int64_t kMaxUnrolledPairBody = 512;

/// Byte budget for the interleaved pair buffer, matching what
/// `promote-buffers-to-stack` keeps on the stack. A heap-resident buffer hides
/// its alignment from the backend, and the unmerged lane loads then cost more
/// than the pairing saves, so a larger shape keeps the single-lane schedule.
constexpr int64_t kMaxPairBufferBytes = 1024;

/// The lane count every web this pass produces carries.
constexpr int64_t kLanes = 2;

/// One single-lane ordered reduction from its zeroed accumulator to the store
/// of its exported sample, plus the policy attributes the paired web reuses
/// verbatim. Every field is required, so a partially understood chain never
/// reaches the rewrite.
struct ReductionWeb {
  ondrix::ondsp::AccZeroOp zero;
  ondrix::ondsp::ReduceMacOp reduce;
  ondrix::ondsp::AccExportOp exportOp;
  memref::StoreOp store;
  ondrix::ondsp::AccType accumulator;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::ProductAttr product;
  int64_t length = 0;
};

/// Static extent of a rank-1 memref whose element type is `storage`.
std::optional<int64_t> getStreamLength(Value value, Type storage) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0) || type.getElementType() != storage)
    return std::nullopt;
  return type.getDimSize(0);
}

/// Matches the four-operation web rooted at a zeroed single-lane accumulator.
/// The chain must be single use throughout and confined to one block, and the
/// export must land in the destination's scalar storage — which is what
/// excludes the raw-widening export form.
FailureOr<ReductionWeb> matchReductionWeb(ondrix::ondsp::AccZeroOp zero) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(zero.getAcc().getType());
  if (!accumulator || !ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return failure();
  if (!zero.getAcc().hasOneUse())
    return failure();

  auto reduce = dyn_cast<ondrix::ondsp::ReduceMacOp>(*zero.getAcc().getUsers().begin());
  if (!reduce || reduce.getInitial() != zero.getAcc() || !reduce.getResult().hasOneUse())
    return failure();
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(reduce.getNumeric());
  if (!numeric || !reduce.getProduct())
    return failure();

  auto exportOp = dyn_cast<ondrix::ondsp::AccExportOp>(*reduce.getResult().getUsers().begin());
  if (!exportOp || exportOp.getAcc() != reduce.getResult() || !exportOp.getResult().hasOneUse())
    return failure();
  if (exportOp.getResult().getType() != exportOp.getDst().getStorage())
    return failure();

  auto store = dyn_cast<memref::StoreOp>(*exportOp.getResult().getUsers().begin());
  if (!store || store.getValueToStore() != exportOp.getResult())
    return failure();

  Block *block = zero->getBlock();
  if (reduce->getBlock() != block || exportOp->getBlock() != block || store->getBlock() != block)
    return failure();

  std::optional<int64_t> lhsLength = getStreamLength(reduce.getLhs(), numeric.getStorage());
  std::optional<int64_t> rhsLength = getStreamLength(reduce.getRhs(), numeric.getStorage());
  if (!lhsLength || !rhsLength || *lhsLength != *rhsLength || *lhsLength <= 0)
    return failure();

  ReductionWeb web;
  web.zero = zero;
  web.reduce = reduce;
  web.exportOp = exportOp;
  web.store = store;
  web.accumulator = accumulator;
  web.numeric = numeric;
  web.product = *reduce.getProduct();
  web.length = *lhsLength;
  return web;
}

/// Whether two webs declare the identical accumulator, product, and export
/// policy over the same length. Only then does one dual-lane web carry both.
bool haveIdenticalPolicy(ReductionWeb first, ReductionWeb second) {
  return first.accumulator == second.accumulator && first.numeric == second.numeric &&
         first.product == second.product && first.length == second.length &&
         first.exportOp.getDst() == second.exportOp.getDst() &&
         first.exportOp.getRounding() == second.exportOp.getRounding() &&
         first.exportOp.getOverflow() == second.exportOp.getOverflow();
}

/// True when the shared stream is the reduction's lhs, false when it is the
/// rhs, and nothing when the two webs share neither side or both.
std::optional<bool> getSharedSide(ReductionWeb first, ReductionWeb second) {
  bool sharedLhs = first.reduce.getLhs() == second.reduce.getLhs();
  bool sharedRhs = first.reduce.getRhs() == second.reduce.getRhs();
  if (sharedLhs == sharedRhs)
    return std::nullopt;
  return sharedLhs;
}

Value getSharedStream(ReductionWeb web, bool sharedLhs) {
  return sharedLhs ? web.reduce.getLhs() : web.reduce.getRhs();
}

Value getDifferingStream(ReductionWeb web, bool sharedLhs) {
  return sharedLhs ? web.reduce.getRhs() : web.reduce.getLhs();
}

/// An index carried alongside the literal it is known to equal, so an unrolled
/// chain folds its own address arithmetic instead of leaving every `2k + 1` to
/// a later canonicalization.
struct FoldedIndex {
  Value value;
  std::optional<int64_t> literal;
};

FoldedIndex getIndexLiteral(OpBuilder &builder, Location loc, int64_t literal) {
  return {builder.create<arith::ConstantIndexOp>(loc, literal), literal};
}

/// Index `base * factor + offset`.
FoldedIndex getScaledIndex(OpBuilder &builder, Location loc, FoldedIndex base, int64_t factor,
                           int64_t offset) {
  if (base.literal)
    return getIndexLiteral(builder, loc, *base.literal * factor + offset);
  Value scaled = base.value;
  if (factor != 1)
    scaled = builder.create<arith::MulIOp>(loc, scaled,
                                           builder.create<arith::ConstantIndexOp>(loc, factor));
  if (offset == 0)
    return {scaled, std::nullopt};
  return {builder.create<arith::AddIOp>(loc, scaled,
                                        builder.create<arith::ConstantIndexOp>(loc, offset)),
          std::nullopt};
}

/// The two lane element indices of one step: adjacent slots `2k` and `2k + 1`
/// of an interleaved buffer, which is what the pair alignment pays for.
std::pair<Value, Value> getLaneSlots(OpBuilder &builder, Location loc, FoldedIndex index) {
  FoldedIndex low = getScaledIndex(builder, loc, index, kLanes, 0);
  return {low.value, getScaledIndex(builder, loc, low, 1, 1).value};
}

/// The two lane slots of one step of output pair `pair` in a flat interleaved
/// pair buffer whose per-pair span is `stride`. Both fold to literals when the
/// pair walk and the chain are unrolled, which is what leaves the backend a
/// constant offset from the buffer base to fold into the load.
std::pair<Value, Value> getPairBufferSlots(OpBuilder &builder, Location loc, FoldedIndex pair,
                                           int64_t stride, FoldedIndex index) {
  FoldedIndex low;
  if (pair.literal && index.literal) {
    low = getIndexLiteral(builder, loc, *pair.literal * stride + *index.literal * kLanes);
  } else {
    FoldedIndex base = getScaledIndex(builder, loc, pair, stride, 0);
    FoldedIndex step = getScaledIndex(builder, loc, index, kLanes, 0);
    low = {builder.create<arith::AddIOp>(loc, base.value, step.value), std::nullopt};
  }
  return {low.value, getScaledIndex(builder, loc, low, 1, 1).value};
}

/// Supplies the two lane values for one index of the paired chain.
using LaneEmitter = llvm::function_ref<std::pair<Value, Value>(OpBuilder &, Location, FoldedIndex)>;

/// Emits the dual-lane web for one output pair and returns its exported
/// `vector<2 x dst>`. The shared stream is loaded as the mac's scalar
/// coefficient and the lane values enter as the vector operand, so both lanes
/// visit the indices in the order the ordered reduction declared.
Value emitPairedWeb(OpBuilder &builder, Location loc, ReductionWeb web, Value shared,
                    LaneEmitter emitLanes) {
  MLIRContext *context = builder.getContext();
  auto laneAccumulator = ondrix::ondsp::AccType::get(
      context, web.accumulator.getStorage(), web.accumulator.getFrac(),
      web.accumulator.getSignedness(), web.accumulator.getUpdateOverflow(),
      static_cast<unsigned>(kLanes));
  auto valueType = VectorType::get({kLanes}, web.numeric.getStorage());

  auto emitStep = [&](OpBuilder &stepBuilder, Location stepLoc, Value accumulator,
                      FoldedIndex index) {
    Value coefficient =
        stepBuilder.create<memref::LoadOp>(stepLoc, shared, ValueRange{index.value});
    std::pair<Value, Value> lanes = emitLanes(stepBuilder, stepLoc, index);
    Value values =
        stepBuilder.create<arith::ConstantOp>(stepLoc, stepBuilder.getZeroAttr(valueType));
    values = stepBuilder.create<vector::InsertOp>(stepLoc, lanes.first, values, 0);
    values = stepBuilder.create<vector::InsertOp>(stepLoc, lanes.second, values, 1);
    return stepBuilder
        .create<ondrix::ondsp::MacOp>(stepLoc, laneAccumulator, accumulator, values, coefficient,
                                      web.numeric, web.product)
        .getResult();
  };

  Value accumulator = builder.create<ondrix::ondsp::AccZeroOp>(loc, laneAccumulator);
  if (web.length <= kMaxUnrolledLength) {
    for (int64_t index = 0; index < web.length; ++index)
      accumulator = emitStep(builder, loc, accumulator, getIndexLiteral(builder, loc, index));
  } else {
    Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value upper = builder.create<arith::ConstantIndexOp>(loc, web.length);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = builder.create<scf::ForOp>(
        loc, lower, upper, step, ValueRange{accumulator},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value index, ValueRange iterArgs) {
          bodyBuilder.create<scf::YieldOp>(
              bodyLoc, emitStep(bodyBuilder, bodyLoc, iterArgs.front(), {index, std::nullopt}));
        });
    accumulator = loop.getResult(0);
  }

  auto exportType = VectorType::get({kLanes}, web.exportOp.getDst().getStorage());
  return builder.create<ondrix::ondsp::AccExportOp>(
      loc, exportType, accumulator, web.exportOp.getDst(), web.exportOp.getRounding(),
      web.exportOp.getOverflow());
}

/// Stores lane `lane` of a paired export at the target the single-lane web it
/// replaces used, with `indices` substituted for that web's store indices.
void storeLane(OpBuilder &builder, Location loc, Value exported, int64_t lane, Value destination,
               ValueRange indices) {
  Value sample = builder.create<vector::ExtractOp>(loc, exported, lane);
  builder.create<memref::StoreOp>(loc, sample, destination, indices);
}

void eraseWeb(ReductionWeb web) {
  web.store.erase();
  web.exportOp.erase();
  web.reduce.erase();
  web.zero.erase();
}

//===----------------------------------------------------------------------===//
// Straight-line pairs with constant differing streams
//===----------------------------------------------------------------------===//

/// The private constant global a differing stream reads, when it reads one
/// directly and its initializer is a dense integer table.
memref::GlobalOp getConstantTable(Value stream, ModuleOp module) {
  auto read = stream.getDefiningOp<memref::GetGlobalOp>();
  if (!read)
    return nullptr;
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(module, read.getNameAttr());
  if (!global || global->getParentOp() != module || !global.getConstant() ||
      SymbolTable::getSymbolVisibility(global) != SymbolTable::Visibility::Private)
    return nullptr;
  if (!isa_and_nonnull<DenseIntElementsAttr>(global.getInitialValueAttr()))
    return nullptr;
  return global;
}

/// Reads the interleaved pair table for two row globals, creating it once per
/// module. A symbol already holding that name is reused only when it is
/// indistinguishable from what this would emit; anything else fails closed and
/// returns nothing, which refuses the pair.
Value getOrCreatePairTable(OpBuilder &builder, Location loc, ModuleOp module, memref::GlobalOp low,
                           memref::GlobalOp high) {
  auto elementType = cast<MemRefType>(low.getType()).getElementType();
  int64_t length = cast<MemRefType>(low.getType()).getDimSize(0);
  auto tableType = MemRefType::get({kLanes * length}, elementType);
  StringRef highSuffix = high.getSymName().rsplit('_').second;
  if (highSuffix.empty())
    highSuffix = high.getSymName();
  std::string symbol = (low.getSymName() + "_" + highSuffix + "_pair").str();

  SmallVector<APInt> lowValues =
      llvm::to_vector(cast<DenseIntElementsAttr>(low.getInitialValueAttr()).getValues<APInt>());
  SmallVector<APInt> highValues =
      llvm::to_vector(cast<DenseIntElementsAttr>(high.getInitialValueAttr()).getValues<APInt>());
  SmallVector<APInt> interleaved;
  interleaved.reserve(kLanes * length);
  for (int64_t index = 0; index < length; ++index) {
    interleaved.push_back(lowValues[index]);
    interleaved.push_back(highValues[index]);
  }
  auto initializer =
      DenseIntElementsAttr::get(RankedTensorType::get({kLanes * length}, elementType), interleaved);
  auto alignment = builder.getI64IntegerAttr(4);

  if (Operation *existing = SymbolTable::lookupSymbolIn(module, symbol)) {
    auto global = dyn_cast<memref::GlobalOp>(existing);
    if (!global || !global.getConstant() || global.getType() != tableType ||
        SymbolTable::getSymbolVisibility(existing) != SymbolTable::Visibility::Private ||
        global.getInitialValueAttr() != initializer || global.getAlignmentAttr() != alignment)
      return nullptr;
    return builder.create<memref::GetGlobalOp>(loc, tableType, symbol);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(low);
    builder.create<memref::GlobalOp>(loc, symbol, builder.getStringAttr("private"), tableType,
                                     initializer, /*constant=*/true, alignment);
  }
  return builder.create<memref::GetGlobalOp>(loc, tableType, symbol);
}

/// Whether every operation the paired web must move across carries no memory
/// effect. The span runs from the first web's accumulator to the second web's
/// store, and both webs' own operations are the ones being reordered.
bool isMotionSpanEffectFree(ReductionWeb first, ReductionWeb second) {
  DenseSet<Operation *> webOperations{first.zero,  first.reduce,  first.exportOp,  first.store,
                                      second.zero, second.reduce, second.exportOp, second.store};
  for (Operation *op = first.zero->getNextNode(); op && op != second.store.getOperation();
       op = op->getNextNode()) {
    if (!webOperations.contains(op) && !isMemoryEffectFree(op))
      return false;
  }
  return true;
}

/// Whether `value` is already defined at `insertionPoint`. A value produced in
/// an enclosing region always is; one produced in the same block must come
/// first. Anything the block relation cannot place is refused.
bool isAvailableAt(Value value, Operation *insertionPoint) {
  Operation *ancestor = value.getParentBlock()->findAncestorOpInBlock(*insertionPoint);
  if (!ancestor)
    return false;
  Operation *producer = value.getDefiningOp();
  return !producer || producer->isBeforeInBlock(ancestor);
}

/// Whether a value created immediately after `anchor` reaches `use`.
bool isAvailableAfter(Operation *anchor, Operation *use) {
  Operation *ancestor = anchor->getBlock()->findAncestorOpInBlock(*use);
  return ancestor && anchor->isBeforeInBlock(ancestor);
}

/// Rewrites one consecutive pair of constant-table webs. Returns the two row
/// globals the pair consumed when it fired, so the module can drop them once
/// nothing reads them.
FailureOr<std::pair<memref::GlobalOp, memref::GlobalOp>>
pairConstantTableWebs(ModuleOp module, ReductionWeb first, ReductionWeb second) {
  if (!haveIdenticalPolicy(first, second))
    return failure();
  std::optional<bool> sharedLhs = getSharedSide(first, second);
  if (!sharedLhs)
    return failure();
  if (!first.store->isBeforeInBlock(second.store))
    return failure();

  Value shared = getSharedStream(first, *sharedLhs);
  Value lowStream = getDifferingStream(first, *sharedLhs);
  Value highStream = getDifferingStream(second, *sharedLhs);
  memref::GlobalOp lowTable = getConstantTable(lowStream, module);
  memref::GlobalOp highTable = getConstantTable(highStream, module);
  if (!lowTable || !highTable || lowTable.getType() != highTable.getType())
    return failure();

  if (!isMotionSpanEffectFree(first, second))
    return failure();
  // The first web's store now happens after the second web's loads, so it must
  // be provably distinct storage from every source the paired web reads.
  Value destination = first.store.getMemRef();
  if (ondrix::conversion::mayShareStorage(destination, shared) ||
      ondrix::conversion::mayShareStorage(destination, highStream))
    return failure();

  SmallVector<Value> required{shared, destination, second.store.getMemRef()};
  llvm::append_range(required, first.store.getIndices());
  llvm::append_range(required, second.store.getIndices());
  if (!llvm::all_of(required, [&](Value value) { return isAvailableAt(value, first.zero); }))
    return failure();

  OpBuilder builder(first.zero);
  Location loc = first.zero.getLoc();
  Value pairTable = getOrCreatePairTable(builder, loc, module, lowTable, highTable);
  if (!pairTable)
    return failure();

  Value exported = emitPairedWeb(
      builder, loc, first, shared,
      [&](OpBuilder &stepBuilder, Location stepLoc, FoldedIndex index) {
        auto [low, high] = getLaneSlots(stepBuilder, stepLoc, index);
        Value lowValue = stepBuilder.create<memref::LoadOp>(stepLoc, pairTable, ValueRange{low});
        Value highValue = stepBuilder.create<memref::LoadOp>(stepLoc, pairTable, ValueRange{high});
        return std::make_pair(lowValue, highValue);
      });
  storeLane(builder, loc, exported, 0, destination, first.store.getIndices());
  storeLane(builder, loc, exported, 1, second.store.getMemRef(), second.store.getIndices());

  Operation *lowRead = lowStream.getDefiningOp();
  Operation *highRead = highStream.getDefiningOp();
  eraseWeb(second);
  eraseWeb(first);
  if (lowRead->use_empty())
    lowRead->erase();
  if (highRead->use_empty())
    highRead->erase();
  return std::make_pair(lowTable, highTable);
}

/// Pairs consecutive compatible webs of one block greedily. A web that fails to
/// pair with its successor is skipped rather than retried against a later one:
/// the pairing is a schedule choice on adjacent outputs, and an unpaired
/// leftover keeps the single-lane schedule.
void pairStraightLineWebs(ModuleOp module, Block &block, SetVector<Operation *> &consumedTables) {
  SmallVector<ReductionWeb> webs;
  for (Operation &op : block) {
    auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(&op);
    if (!zero)
      continue;
    FailureOr<ReductionWeb> web = matchReductionWeb(zero);
    if (succeeded(web))
      webs.push_back(*web);
  }

  for (size_t index = 0; index + 1 < webs.size();) {
    FailureOr<std::pair<memref::GlobalOp, memref::GlobalOp>> tables =
        pairConstantTableWebs(module, webs[index], webs[index + 1]);
    if (failed(tables)) {
      ++index;
      continue;
    }
    consumedTables.insert(tables->first);
    consumedTables.insert(tables->second);
    index += 2;
  }
}

//===----------------------------------------------------------------------===//
// Transposed-pack column loops
//===----------------------------------------------------------------------===//

/// A loop over `[0, extent)` with unit step and no iteration arguments.
bool isUnitStepLoop(scf::ForOp loop, int64_t extent) {
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  return loop.getInitArgs().empty() && lower && upper && step && *lower == 0 && *step == 1 &&
         upper == extent;
}

bool hasIndices(ValueRange indices, ValueRange expected) { return llvm::equal(indices, expected); }

/// Operations of a loop body that carry the shape, with constants skipped.
SmallVector<Operation *> getShapeOperations(Block &body) {
  SmallVector<Operation *> operations;
  for (Operation &op : body.without_terminator()) {
    if (matchPattern(&op, m_Constant()))
      continue;
    operations.push_back(&op);
  }
  return operations;
}

/// Matches the two perfectly nested loops that transpose `src[k][c]` into
/// `packed[c][k]`, and returns the source. Any other body shape leaves the pack
/// buffer unrecognized.
Value matchPackNest(scf::ForOp outer, Value packed) {
  auto packedType = cast<MemRefType>(packed.getType());
  if (!isUnitStepLoop(outer, packedType.getDimSize(0)))
    return nullptr;
  SmallVector<Operation *> outerBody = getShapeOperations(*outer.getBody());
  if (outerBody.size() != 1)
    return nullptr;
  auto inner = dyn_cast<scf::ForOp>(outerBody.front());
  if (!inner || !isUnitStepLoop(inner, packedType.getDimSize(1)))
    return nullptr;

  SmallVector<Operation *> innerBody = getShapeOperations(*inner.getBody());
  if (innerBody.size() != 2)
    return nullptr;
  auto load = dyn_cast<memref::LoadOp>(innerBody[0]);
  auto store = dyn_cast<memref::StoreOp>(innerBody[1]);
  if (!load || !store || store.getValueToStore() != load.getResult())
    return nullptr;

  Value column = outer.getInductionVar();
  Value row = inner.getInductionVar();
  if (!hasIndices(load.getIndices(), {row, column}) || store.getMemRef() != packed ||
      !hasIndices(store.getIndices(), {column, row}))
    return nullptr;
  auto sourceType = dyn_cast<MemRefType>(load.getMemRef().getType());
  if (!sourceType || sourceType.getRank() != 2 ||
      sourceType.getElementType() != packedType.getElementType())
    return nullptr;
  return load.getMemRef();
}

/// One column loop the pack buffer feeds: the row view it takes, the web it
/// runs, and where the loop's induction variable appears in the store.
struct ColumnLoop {
  scf::ForOp loop;
  memref::SubViewOp view;
  ReductionWeb web;
  Value shared;
  bool sharedLhs = false;
  unsigned columnIndex = 0;
};

/// Matches a loop whose body is exactly one row view of the pack buffer and one
/// candidate web over it, with the store selecting its column by the loop's
/// induction variable alone.
FailureOr<ColumnLoop> matchColumnLoop(memref::SubViewOp view, Value packed) {
  auto loop = dyn_cast<scf::ForOp>(view->getParentOp());
  auto packedType = cast<MemRefType>(packed.getType());
  if (!loop || !isUnitStepLoop(loop, packedType.getDimSize(0)))
    return failure();

  SmallVector<Operation *> body = getShapeOperations(*loop.getBody());
  if (body.size() != 5 || body[0] != view.getOperation())
    return failure();
  auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(body[1]);
  if (!zero)
    return failure();
  FailureOr<ReductionWeb> web = matchReductionWeb(zero);
  if (failed(web) || web->reduce != body[2] || web->exportOp != body[3] || web->store != body[4])
    return failure();

  Value column = loop.getInductionVar();
  if (view.getSource() != packed || view.getType().getRank() != 1)
    return failure();
  SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = view.getMixedSizes();
  SmallVector<OpFoldResult> strides = view.getMixedStrides();
  if (offsets.size() != 2 || offsets[0].dyn_cast<Value>() != column ||
      getConstantIntValue(offsets[1]) != 0)
    return failure();
  if (sizes.size() != 2 || getConstantIntValue(sizes[0]) != 1 ||
      getConstantIntValue(sizes[1]) != packedType.getDimSize(1) || strides.size() != 2 ||
      getConstantIntValue(strides[0]) != 1 || getConstantIntValue(strides[1]) != 1)
    return failure();

  ColumnLoop match;
  match.loop = loop;
  match.view = view;
  match.web = *web;
  if (web->reduce.getLhs() == view.getResult())
    match.sharedLhs = false;
  else if (web->reduce.getRhs() == view.getResult())
    match.sharedLhs = true;
  else
    return failure();
  match.shared = getSharedStream(*web, match.sharedLhs);
  if (match.shared.getParentBlock() == loop.getBody())
    return failure();

  ValueRange indices = web->store.getIndices();
  std::optional<unsigned> columnPosition;
  for (auto [position, index] : llvm::enumerate(indices)) {
    if (index == column) {
      if (columnPosition)
        return failure();
      columnPosition = position;
      continue;
    }
    if (index.getParentBlock() == loop.getBody())
      return failure();
  }
  if (!columnPosition)
    return failure();
  match.columnIndex = *columnPosition;

  // The paired web stores the low lane after loading the high lane's column, so
  // the destination must be provably distinct from the shared stream.
  if (ondrix::conversion::mayShareStorage(web->store.getMemRef(), match.shared))
    return failure();
  if (web->store.getMemRef().getParentBlock() == loop.getBody())
    return failure();
  return match;
}

/// Emits the interleaved pack nest beside the transposing one: output columns
/// `2p` and `2p + 1` land adjacent, so one lane pair is one aligned pair of
/// elements. The buffer is a rank-1 `alloca` because the declared alignment
/// reaches the backend only on a stack base; the caller bounds its size against
/// `kMaxPairBufferBytes` and places it in an automatic allocation scope.
Value emitPairPackNest(OpBuilder &builder, Location loc, Value source, MemRefType packedType,
                       int64_t pairs) {
  int64_t rows = packedType.getDimSize(1);
  int64_t stride = kLanes * rows;
  auto pairType = MemRefType::get({pairs * stride}, packedType.getElementType());
  auto alloc = builder.create<memref::AllocaOp>(loc, pairType, builder.getI64IntegerAttr(4));

  Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value pairBound = builder.create<arith::ConstantIndexOp>(loc, pairs);
  Value rowBound = builder.create<arith::ConstantIndexOp>(loc, rows);
  builder.create<scf::ForOp>(
      loc, lower, pairBound, step, ValueRange{},
      [&](OpBuilder &pairBuilder, Location pairLoc, Value pair, ValueRange) {
        pairBuilder.create<scf::ForOp>(
            pairLoc, lower, rowBound, step, ValueRange{},
            [&](OpBuilder &rowBuilder, Location rowLoc, Value row, ValueRange) {
              auto [lowColumn, highColumn] = getLaneSlots(rowBuilder, rowLoc, {pair, std::nullopt});
              auto [lowSlot, highSlot] = getPairBufferSlots(
                  rowBuilder, rowLoc, {pair, std::nullopt}, stride, {row, std::nullopt});
              Value columns[] = {lowColumn, highColumn};
              Value slots[] = {lowSlot, highSlot};
              for (int64_t lane = 0; lane < kLanes; ++lane) {
                Value element = rowBuilder.create<memref::LoadOp>(rowLoc, source,
                                                                  ValueRange{row, columns[lane]});
                rowBuilder.create<memref::StoreOp>(rowLoc, element, alloc.getResult(),
                                                   ValueRange{slots[lane]});
              }
              rowBuilder.create<scf::YieldOp>(rowLoc);
            });
        pairBuilder.create<scf::YieldOp>(pairLoc);
      });
  return alloc.getResult();
}

/// Emits one output pair of a rewritten column loop: the dual-lane web over
/// pair `pair`, then the two stores at columns `2p` and `2p + 1`.
void emitColumnPair(OpBuilder &builder, Location loc, ColumnLoop match, Value pairPacked,
                    int64_t stride, FoldedIndex pair) {
  Value exported = emitPairedWeb(
      builder, loc, match.web, match.shared,
      [&](OpBuilder &stepBuilder, Location stepLoc, FoldedIndex index) {
        auto [low, high] = getPairBufferSlots(stepBuilder, stepLoc, pair, stride, index);
        Value lowValue = stepBuilder.create<memref::LoadOp>(stepLoc, pairPacked, ValueRange{low});
        Value highValue = stepBuilder.create<memref::LoadOp>(stepLoc, pairPacked, ValueRange{high});
        return std::make_pair(lowValue, highValue);
      });

  auto [lowColumn, highColumn] = getLaneSlots(builder, loc, pair);
  Value columns[] = {lowColumn, highColumn};
  for (int64_t lane = 0; lane < kLanes; ++lane) {
    SmallVector<Value> indices(match.web.store.getIndices());
    indices[match.columnIndex] = columns[lane];
    storeLane(builder, loc, exported, lane, match.web.store.getMemRef(), indices);
  }
}

/// Replaces a matched column loop with a walk over output pairs. The pair walk
/// is unrolled only while the whole body stays inside the straight-line bound.
void rewriteColumnLoop(ColumnLoop match, Value pairPacked, int64_t stride, int64_t pairs) {
  OpBuilder builder(match.loop);
  Location loc = match.loop.getLoc();
  if (pairs * match.web.length <= kMaxUnrolledPairBody) {
    for (int64_t pair = 0; pair < pairs; ++pair)
      emitColumnPair(builder, loc, match, pairPacked, stride, getIndexLiteral(builder, loc, pair));
  } else {
    Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value upper = builder.create<arith::ConstantIndexOp>(loc, pairs);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    builder.create<scf::ForOp>(
        loc, lower, upper, step, ValueRange{},
        [&](OpBuilder &pairBuilder, Location pairLoc, Value pair, ValueRange) {
          emitColumnPair(pairBuilder, pairLoc, match, pairPacked, stride, {pair, std::nullopt});
          pairBuilder.create<scf::YieldOp>(pairLoc);
        });
  }
  match.loop.erase();
}

/// Rewrites the column loops one transposed pack buffer feeds. The buffer is
/// recognized only when every user is a pack-nest store, a column-loop row
/// view, or the single dealloc, so a user this cannot classify leaves the whole
/// allocation on the ordered schedule.
void pairPackedColumnLoops(memref::AllocOp alloc) {
  auto packedType = alloc.getType();
  if (packedType.getRank() != 2 || !packedType.hasStaticShape())
    return;
  int64_t columns = packedType.getDimSize(0);
  int64_t rows = packedType.getDimSize(1);
  if (columns <= 0 || columns % kLanes != 0 || rows <= 0)
    return;

  // The interleaved buffer holds the same elements as the transposed one, so
  // its byte size is `columns * rows * elementBytes`. Either dimension past the
  // budget already exceeds it, which also keeps the product from overflowing.
  if (!packedType.getElementType().isIntOrFloat() || columns > kMaxPairBufferBytes ||
      rows > kMaxPairBufferBytes)
    return;
  int64_t elementBytes = llvm::divideCeil(packedType.getElementTypeBitWidth(), 8);
  if (columns * rows * elementBytes > kMaxPairBufferBytes)
    return;

  memref::StoreOp packStore;
  memref::DeallocOp dealloc;
  SmallVector<memref::SubViewOp> views;
  for (Operation *user : alloc->getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (packStore || store.getMemRef() != alloc.getResult())
        return;
      packStore = store;
      continue;
    }
    if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      views.push_back(view);
      continue;
    }
    if (auto free = dyn_cast<memref::DeallocOp>(user)) {
      if (dealloc)
        return;
      dealloc = free;
      continue;
    }
    return;
  }
  if (!packStore || !dealloc || views.empty())
    return;

  auto inner = dyn_cast<scf::ForOp>(packStore->getParentOp());
  auto outer = inner ? dyn_cast<scf::ForOp>(inner->getParentOp()) : nullptr;
  if (!outer)
    return;
  Value source = matchPackNest(outer, alloc.getResult());
  if (!source)
    return;
  // The pair buffer is a stack allocation, so it must land directly in an
  // automatic allocation scope rather than inside a loop that would repeat it.
  if (!outer->getBlock()->getParentOp()->hasTrait<OpTrait::AutomaticAllocationScope>())
    return;

  // The interleaved buffer is materialized right after the transposing nest, so
  // every column loop it feeds must be reachable from there.
  SmallVector<ColumnLoop> matches;
  for (memref::SubViewOp view : views) {
    FailureOr<ColumnLoop> match = matchColumnLoop(view, alloc.getResult());
    if (failed(match) || !isAvailableAfter(outer, match->loop))
      return;
    matches.push_back(*match);
  }

  OpBuilder builder(outer->getBlock(), std::next(outer->getIterator()));
  Value pairPacked =
      emitPairPackNest(builder, outer.getLoc(), source, packedType, columns / kLanes);

  for (ColumnLoop match : matches)
    rewriteColumnLoop(match, pairPacked, kLanes * rows, columns / kLanes);
  dealloc.erase();
  outer.erase();
  alloc.erase();
}

class PairOndspFixedReductionOutputsPass final
    : public ondrix::impl::PairOndspFixedReductionOutputsBase<PairOndspFixedReductionOutputsPass> {
public:
  using ondrix::impl::PairOndspFixedReductionOutputsBase<
      PairOndspFixedReductionOutputsPass>::PairOndspFixedReductionOutputsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect first: both matchers erase the operations they consume, and the
    // column-loop rewrite creates allocations the pack matcher must never see.
    SmallVector<memref::AllocOp> allocations;
    module.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
    for (memref::AllocOp alloc : allocations)
      pairPackedColumnLoops(alloc);

    SmallVector<Block *> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });
    SetVector<Operation *> consumedTables;
    for (Block *block : blocks)
      pairStraightLineWebs(module, *block, consumedTables);

    for (Operation *table : consumedTables) {
      if (SymbolTable::symbolKnownUseEmpty(table, module))
        table->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createPairOndspFixedReductionOutputsPass() {
  return std::make_unique<PairOndspFixedReductionOutputsPass>();
}
