#include "ondrix/Conversion/OndspToOrtumCore/OndspToOrtumCore.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class ConvertOndspToOrtumCorePass
    : public PassWrapper<ConvertOndspToOrtumCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOndspToOrtumCorePass)

  StringRef getArgument() const final { return "convert-ondsp-to-ortumcore"; }
  StringRef getDescription() const final {
    return "Lower ondsp semantic ops to ortumcore target semantic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ondrix::ondsp::OndspDialect, ondrix::ortumcore::OrtumCoreDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "ondsp.mac" || name == "ondsp.reduce_mac" || name == "ondsp.cx_butterfly")
        worklist.push_back(op);
    });

    OpBuilder builder(module.getContext());
    for (Operation *op : worklist) {
      builder.setInsertionPoint(op);
      StringRef name = op->getName().getStringRef();
      if (name == "ondsp.cx_butterfly") {
        if (failed(lowerButterfly(builder, op))) {
          signalPassFailure();
          return;
        }
        continue;
      }
      if (name == "ondsp.reduce_mac") {
        if (failed(lowerReduceMac(builder, op))) {
          signalPassFailure();
          return;
        }
        continue;
      }

      FailureOr<StringRef> target = chooseMacTarget(op);
      if (failed(target)) {
        op->emitError("unsupported ondsp numeric policy for ortumcore MAC lowering");
        signalPassFailure();
        return;
      }

      OperationState state(op->getLoc(), *target);
      state.addOperands(op->getOperands());
      state.addTypes(op->getResultTypes());
      Operation *replacement = builder.create(state);
      op->replaceAllUsesWith(replacement);
      op->erase();
    }
  }

private:
  static FailureOr<StringRef> chooseMacTarget(Operation *op) {
    auto fixed = dyn_cast_or_null<ondrix::ondsp::FixedAttr>(op->getAttr("numeric"));
    if (!fixed)
      return failure();

    auto intType = fixed.getStorage().dyn_cast<IntegerType>();
    if (!intType)
      return failure();

    if (intType.getWidth() == 16)
      return StringRef("ortumcore.dual_mac");
    if (intType.getWidth() == 32)
      return StringRef("ortumcore.qmac_add");
    return failure();
  }

  static LogicalResult lowerReduceMac(OpBuilder &builder, Operation *op) {
    FailureOr<StringRef> target = chooseMacTarget(op);
    if (failed(target)) {
      op->emitError("unsupported ondsp numeric policy for ortumcore reduce_mac lowering");
      return failure();
    }

    Type accType = ondrix::ortumcore::AccumType::get(builder.getContext());

    OperationState initState(op->getLoc(), "ortumcore.acc_init");
    initState.addTypes(accType);
    Operation *init = builder.create(initState);

    OperationState macState(op->getLoc(), *target);
    macState.addOperands({init->getResult(0), op->getOperand(0), op->getOperand(1)});
    macState.addTypes(accType);
    Operation *mac = builder.create(macState);

    OperationState extractState(op->getLoc(), "ortumcore.acc_extract");
    extractState.addOperands(mac->getResult(0));
    extractState.addTypes(op->getResultTypes());
    Operation *extract = builder.create(extractState);

    op->replaceAllUsesWith(extract);
    op->erase();
    return success();
  }

  static LogicalResult lowerButterfly(OpBuilder &builder, Operation *op) {
    auto fixed = dyn_cast_or_null<ondrix::ondsp::FixedAttr>(op->getAttr("numeric"));
    if (!fixed) {
      op->emitError("expected fixed numeric policy for ortumcore butterfly lowering");
      return failure();
    }

    auto intType = fixed.getStorage().dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() != 16) {
      op->emitError("only packed q15 butterfly lowering is supported");
      return failure();
    }

    auto layout = dyn_cast_or_null<ondrix::ondsp::CxLayoutAttr>(op->getAttr("layout"));
    if (!layout) {
      op->emitError("expected explicit complex layout for ortumcore butterfly lowering");
      return failure();
    }
    auto layoutValue = layout.getLayout();
    if (layoutValue != ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo &&
        layoutValue != ondrix::ondsp::ComplexLayout::PackedI16RealHiImagLo) {
      op->emitError("only packed i16 complex layouts are supported");
      return failure();
    }

    Type stateType = ondrix::ortumcore::VecStateType::get(builder.getContext());

    OperationState initState(op->getLoc(), "ortumcore.vec_state_init");
    initState.addTypes(stateType);
    Operation *init = builder.create(initState);

    OperationState modeState(op->getLoc(), "ortumcore.vec_set_mode");
    modeState.addOperands(init->getResult(0));
    modeState.addAttribute("sat", builder.getBoolAttr(true));
    modeState.addAttribute("rnd", builder.getBoolAttr(false));
    modeState.addAttribute("cpack", builder.getBoolAttr(true));
    modeState.addAttribute("shiftr", builder.getI64IntegerAttr(15));
    modeState.addAttribute("shiftl", builder.getI64IntegerAttr(0));
    modeState.addTypes(stateType);
    Operation *mode = builder.create(modeState);

    auto trivialTwiddle = dyn_cast_or_null<BoolAttr>(op->getAttr("trivial_twiddle"));
    if (!trivialTwiddle || !trivialTwiddle.getValue())
      return lowerGeneralButterfly(builder, op, stateType, mode->getResult(0));

    OperationState fftState(op->getLoc(), "ortumcore.fft_primitive_7");
    fftState.addOperands({mode->getResult(0), op->getOperand(0), op->getOperand(1)});
    fftState.addTypes({stateType, op->getResult(0).getType(), op->getResult(1).getType()});
    Operation *fft = builder.create(fftState);

    op->getResult(0).replaceAllUsesWith(fft->getResult(1));
    op->getResult(1).replaceAllUsesWith(fft->getResult(2));
    op->erase();
    return success();
  }

  static LogicalResult lowerGeneralButterfly(OpBuilder &builder, Operation *op, Type stateType,
                                             Value state) {
    OperationState mulState(op->getLoc(), "ortumcore.cx_mul");
    mulState.addOperands({state, op->getOperand(1), op->getOperand(2)});
    mulState.addTypes({stateType, op->getOperand(1).getType()});
    Operation *mul = builder.create(mulState);

    OperationState addState(op->getLoc(), "ortumcore.cx_dual_add");
    addState.addOperands({mul->getResult(0), op->getOperand(0), mul->getResult(1)});
    addState.addTypes({stateType, op->getResult(0).getType()});
    Operation *add = builder.create(addState);

    OperationState subState(op->getLoc(), "ortumcore.cx_dual_sub");
    subState.addOperands({add->getResult(0), op->getOperand(0), mul->getResult(1)});
    subState.addTypes({stateType, op->getResult(1).getType()});
    Operation *sub = builder.create(subState);

    op->getResult(0).replaceAllUsesWith(add->getResult(1));
    op->getResult(1).replaceAllUsesWith(sub->getResult(1));
    op->erase();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspToOrtumCorePass() {
  return std::make_unique<ConvertOndspToOrtumCorePass>();
}

void ondrix::registerConvertOndspToOrtumCorePass() {
  PassRegistration<ConvertOndspToOrtumCorePass>();
}
