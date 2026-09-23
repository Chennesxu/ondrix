#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_LOWERONDSPASSUMPTIONS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using ondrix::ondsp::AssumeL1BoundOp;

namespace {

/// Sums the absolute raw values in i64 and aborts, with the entry checks'
/// message convention, unless the sum stays below the declared bound.
void emitBoundCheck(AssumeL1BoundOp assume) {
  OpBuilder builder(assume);
  Location loc = assume.getLoc();
  Value source = assume.getSource();
  Type wide = builder.getI64Type();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value length = builder.create<memref::DimOp>(loc, source, 0);
  Value initial = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto loop = builder.create<scf::ForOp>(
      loc, zero, length, one, ValueRange{initial},
      [&](OpBuilder &body, Location bodyLoc, Value index, ValueRange sums) {
        Value element = body.create<memref::LoadOp>(bodyLoc, source, index);
        Value value = body.create<arith::ExtSIOp>(bodyLoc, wide, element);
        Value negated = body.create<arith::SubIOp>(
            bodyLoc, body.create<arith::ConstantIntOp>(bodyLoc, 0, 64), value);
        Value magnitude = body.create<arith::MaxSIOp>(bodyLoc, value, negated);
        body.create<scf::YieldOp>(
            bodyLoc, body.create<arith::AddIOp>(bodyLoc, sums.front(), magnitude).getResult());
      });
  Value bound = builder.create<arith::ConstantIntOp>(loc, assume.getBound(), 64);
  Value holds =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, loop.getResult(0), bound);
  auto function = assume->getParentOfType<func::FuncOp>();
  std::string message = "ondrix_" + (function ? function.getSymName().str() : std::string()) +
                        ": coefficients exceed the declared gain_bound";
  builder.create<cf::AssertOp>(loc, holds, message);
}

struct LowerOndspAssumptions final
    : ondrix::impl::LowerOndspAssumptionsBase<LowerOndspAssumptions> {
  void runOnOperation() override {
    SmallVector<AssumeL1BoundOp> assumptions;
    getOperation()->walk([&](AssumeL1BoundOp op) { assumptions.push_back(op); });
    for (AssumeL1BoundOp assume : assumptions) {
      if (checked)
        emitBoundCheck(assume);
      assume.getResult().replaceAllUsesWith(assume.getSource());
      assume.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createLowerOndspAssumptionsPass() {
  return std::make_unique<LowerOndspAssumptions>();
}
