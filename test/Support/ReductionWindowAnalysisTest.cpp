#include "ondrix/Analysis/ReductionWindowAnalysis.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <optional>

using ondrix::analysis::getStraightLineCarriedWindow;

namespace {

/// One reduction over a window of a signal, in the shape the sliding filters
/// bufferize into: an output loop whose induction variable is the window's
/// own offset.
struct Fixture {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;
  mlir::Type i32;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::AccType accumulator;

  Fixture() : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))), builder(&context) {
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                        ondrix::ondsp::OndspDialect>();
    i32 = builder.getI32Type();
    numeric = ondrix::ondsp::FixedAttr::get(&context, ondrix::ondsp::Signedness::Signed, i32, 31);
    accumulator = ondrix::ondsp::AccType::get(&context, builder.getI64Type(), 62,
                                              ondrix::ondsp::Signedness::Signed,
                                              ondrix::ondsp::OverflowMode::Wrap);
  }

  /// `window` < 0 leaves the signal operand whole, which is the shape of a
  /// reduction that does not slide; `offsetIsLoop` false takes the window at a
  /// constant offset instead of the induction variable.
  ondrix::ondsp::ReduceMacOp build(int64_t taps, int64_t window, int64_t step, int64_t viewStride,
                                   bool inLoop = true, bool offsetIsLoop = true) {
    mlir::Location loc = module->getLoc();
    builder.setInsertionPointToStart(module->getBody());
    auto signalType = mlir::MemRefType::get({1024}, i32);
    auto tapsType = mlir::MemRefType::get({taps}, i32);
    auto function = builder.create<mlir::func::FuncOp>(
        loc, "reduce", builder.getFunctionType({signalType, tapsType}, {}));
    mlir::Block *body = function.addEntryBlock();
    builder.setInsertionPointToStart(body);

    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value bound = builder.create<mlir::arith::ConstantIndexOp>(loc, 64);
    mlir::Value stepValue = builder.create<mlir::arith::ConstantIndexOp>(loc, step);
    mlir::Value initial = builder.create<ondrix::ondsp::AccZeroOp>(loc, accumulator);

    auto emit = [&](mlir::Value offset) {
      mlir::Value signal = body->getArgument(0);
      if (window >= 0)
        signal = builder.create<mlir::memref::SubViewOp>(
            loc, signal, mlir::ArrayRef<mlir::OpFoldResult>{offset},
            mlir::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(window)},
            mlir::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(viewStride)});
      return builder.create<ondrix::ondsp::ReduceMacOp>(loc, accumulator.getStorage(), initial,
                                                        signal, body->getArgument(1), numeric,
                                                        ondrix::ondsp::ProductAttr());
    };

    if (!inLoop) {
      auto reduce = emit(zero);
      builder.setInsertionPointAfter(reduce);
      builder.create<mlir::func::ReturnOp>(loc);
      return reduce;
    }
    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, bound, stepValue);
    builder.setInsertionPointToStart(loop.getBody());
    auto reduce = emit(offsetIsLoop ? loop.getInductionVar() : zero);
    builder.setInsertionPointAfter(loop);
    builder.create<mlir::func::ReturnOp>(loc);
    return reduce;
  }
};

bool expect(std::optional<int64_t> got, std::optional<int64_t> want, const char *what) {
  if (got == want)
    return true;
  llvm::errs() << "reduction window: " << what << " gave "
               << (got ? std::to_string(*got) : std::string("nothing")) << ", wanted "
               << (want ? std::to_string(*want) : std::string("nothing")) << "\n";
  return false;
}

// The straight-line form carries what consecutive windows share, so the
// quantity is the window minus one trip's advance and nothing else.
bool testCarriedWindow() {
  {
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(16, 16, 1, 1)), 15, "16 taps, step 1"))
      return false;
  }
  {
    // A decimating filter advances by its step, so the same window carries less.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(16, 16, 4, 1)), 12, "16 taps, step 4"))
      return false;
  }
  {
    // Past the window length consecutive trips share nothing at all.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(4, 4, 8, 1)), 0, "window shorter than step"))
      return false;
  }
  {
    // A whole-buffer reduction inside a loop reads the same elements every
    // trip, so unrolling it creates no carried value either.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(16, -1, 1, 1)), 0, "no window"))
      return false;
  }
  {
    // The dot shape: no enclosing loop, nothing to carry across.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(64, -1, 1, 1, /*inLoop=*/false)), 0,
                "no enclosing loop"))
      return false;
  }
  {
    // A window at a constant offset does not slide with this loop; the
    // analysis refuses rather than reading it as stationary.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(
                    f.build(16, 16, 1, 1, /*inLoop=*/true, /*offsetIsLoop=*/false)),
                std::nullopt, "offset is not the induction variable"))
      return false;
  }
  {
    // A strided view reads a different element set than its length suggests.
    Fixture f;
    if (!expect(getStraightLineCarriedWindow(f.build(16, 16, 1, 2)), std::nullopt,
                "non-unit view stride"))
      return false;
  }
  return true;
}

} // namespace

int main() {
  if (!testCarriedWindow()) {
    llvm::errs() << "reduction window analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "reduction window analysis: PASS\n";
  return 0;
}
