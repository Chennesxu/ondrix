#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Support/ElementwiseQ15Contract.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_FUSEONDRIXELEMENTWISECHAINS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

using ondrix::certifyUnaryChain;
using ondrix::certifyUnaryChainIsIdentity;
using ondrix::ElementwiseUnaryKind;
using ondrix::ElementwiseUnaryStep;

// The chain the pass reasons about: one unary elementwise operation, read
// back as the contract's own vocabulary so the certificate never sees IR.
struct UnaryLink {
  Operation *op;
  Value input;
  ElementwiseUnaryStep step;
  Attribute numeric;
};

std::optional<UnaryLink> readUnary(Operation *op) {
  if (auto abs = dyn_cast<ondrix::ir::AbsOp>(op))
    return UnaryLink{
        op,
        abs.getInput(),
        {ElementwiseUnaryKind::Abs, 0, ondrix::ondsp::RoundingMode::NearestEven, abs.getOverflow()},
        abs.getNumeric()};
  if (auto negate = dyn_cast<ondrix::ir::NegateOp>(op))
    return UnaryLink{op,
                     negate.getInput(),
                     {ElementwiseUnaryKind::Negate, 0, ondrix::ondsp::RoundingMode::NearestEven,
                      negate.getOverflow()},
                     negate.getNumeric()};
  if (auto offset = dyn_cast<ondrix::ir::OffsetOp>(op))
    return UnaryLink{op,
                     offset.getInput(),
                     {ElementwiseUnaryKind::Offset, offset.getBiasAttr().getInt(),
                      ondrix::ondsp::RoundingMode::NearestEven, offset.getOverflow()},
                     offset.getNumeric()};
  if (auto shift = dyn_cast<ondrix::ir::ShiftOp>(op))
    return UnaryLink{op,
                     shift.getInput(),
                     {ElementwiseUnaryKind::Shift, shift.getAmountAttr().getInt(),
                      shift.getRounding(), shift.getOverflow()},
                     shift.getNumeric()};
  return std::nullopt;
}

// The single operation a chain could collapse to, before any certificate has
// been run. Only shapes the contract can express are proposed: the merged
// parameter has to stay inside the operation's declared range, and the
// merged boundary is the outer one, because that is the boundary the
// program's last observable event already went through.
std::optional<ElementwiseUnaryStep> proposeMerge(const ElementwiseUnaryStep &inner,
                                                 const ElementwiseUnaryStep &outer) {
  if (inner.kind == ElementwiseUnaryKind::Shift && outer.kind == ElementwiseUnaryKind::Shift) {
    int64_t amount = inner.parameter + outer.parameter;
    if (amount < -15 || amount > 15)
      return std::nullopt;
    return ElementwiseUnaryStep{ElementwiseUnaryKind::Shift, amount, outer.rounding,
                                outer.overflow};
  }
  if (inner.kind == ElementwiseUnaryKind::Offset && outer.kind == ElementwiseUnaryKind::Offset) {
    int64_t bias = inner.parameter + outer.parameter;
    if (bias < -32768 || bias > 32767)
      return std::nullopt;
    return ElementwiseUnaryStep{ElementwiseUnaryKind::Offset, bias, outer.rounding, outer.overflow};
  }
  // Anything followed by an absolute value is a candidate absolute value:
  // abs(abs(x)) and abs(negate(x)) both look like abs(x), and whether they
  // are is the certificate's question, not this function's.
  if (outer.kind == ElementwiseUnaryKind::Abs &&
      (inner.kind == ElementwiseUnaryKind::Abs || inner.kind == ElementwiseUnaryKind::Negate))
    return ElementwiseUnaryStep{ElementwiseUnaryKind::Abs, 0, outer.rounding, outer.overflow};
  return std::nullopt;
}

Value buildStep(OpBuilder &builder, Location loc, Type resultType, Value input,
                const ElementwiseUnaryStep &step, Attribute numeric) {
  MLIRContext *context = builder.getContext();
  auto overflow = ondrix::ondsp::OverflowModeAttr::get(context, step.overflow);
  switch (step.kind) {
  case ElementwiseUnaryKind::Abs:
    return builder.create<ondrix::ir::AbsOp>(loc, resultType, input, numeric, overflow);
  case ElementwiseUnaryKind::Negate:
    return builder.create<ondrix::ir::NegateOp>(loc, resultType, input, numeric, overflow);
  case ElementwiseUnaryKind::Offset:
    return builder.create<ondrix::ir::OffsetOp>(
        loc, resultType, input, builder.getI64IntegerAttr(step.parameter), numeric, overflow);
  case ElementwiseUnaryKind::Shift:
    return builder.create<ondrix::ir::ShiftOp>(
        loc, resultType, input, builder.getI64IntegerAttr(step.parameter), numeric,
        ondrix::ondsp::RoundingModeAttr::get(context, step.rounding), overflow);
  }
  return Value();
}

llvm::StringRef describeKind(ElementwiseUnaryKind kind) {
  switch (kind) {
  case ElementwiseUnaryKind::Abs:
    return "abs";
  case ElementwiseUnaryKind::Negate:
    return "negate";
  case ElementwiseUnaryKind::Offset:
    return "offset";
  case ElementwiseUnaryKind::Shift:
    return "shift";
  }
  return "";
}

class FuseOndrixElementwiseChainsPass final
    : public ondrix::impl::FuseOndrixElementwiseChainsBase<FuseOndrixElementwiseChainsPass> {
public:
  using ondrix::impl::FuseOndrixElementwiseChainsBase<
      FuseOndrixElementwiseChainsPass>::FuseOndrixElementwiseChainsBase;

  void runOnOperation() override {
    // Collapsing a pair can expose another pair, so iterate to a fixpoint;
    // every accepted rewrite is certified on its own.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<Operation *> candidates;
      getOperation().walk([&](Operation *op) {
        if (readUnary(op))
          candidates.push_back(op);
      });
      for (Operation *op : candidates) {
        std::optional<UnaryLink> outer = readUnary(op);
        if (!outer)
          continue;
        Operation *innerOp = outer->input.getDefiningOp();
        if (!innerOp || !innerOp->getResult(0).hasOneUse())
          continue;
        std::optional<UnaryLink> inner = readUnary(innerOp);
        if (!inner || inner->numeric != outer->numeric)
          continue;
        // The certificate enumerates the 65536 Q1.15 inputs and has no
        // floating-point analogue.
        if (!isa<ondrix::ondsp::FixedAttr>(outer->numeric))
          continue;

        OpBuilder builder(op);
        Value replacement;
        if (certifyUnaryChainIsIdentity(inner->step, outer->step)) {
          // An identity collapse builds nothing, so there is nothing to
          // stamp: the input's producer was not part of this certificate
          // and must not inherit its provenance.
          replacement = inner->input;
        } else {
          std::optional<ElementwiseUnaryStep> merged = proposeMerge(inner->step, outer->step);
          if (!merged || !certifyUnaryChain(inner->step, outer->step, *merged))
            continue;
          replacement = buildStep(builder, op->getLoc(), op->getResult(0).getType(), inner->input,
                                  *merged, outer->numeric);
          NamedAttrList provenance;
          provenance.append("merged", builder.getStringAttr(describeKind(merged->kind)));
          provenance.append("parameter", builder.getI64IntegerAttr(merged->parameter));
          provenance.append("inner", builder.getStringAttr(describeKind(inner->step.kind)));
          provenance.append("outer", builder.getStringAttr(describeKind(outer->step.kind)));
          provenance.append("exhaustive_inputs", builder.getI64IntegerAttr(65536));
          replacement.getDefiningOp()->setAttr("ondrix.elementwise_fusion_provenance",
                                               provenance.getDictionary(builder.getContext()));
        }
        op->getResult(0).replaceAllUsesWith(replacement);
        op->erase();
        innerOp->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createFuseOndrixElementwiseChainsPass() {
  return std::make_unique<FuseOndrixElementwiseChainsPass>();
}
