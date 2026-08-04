#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/SetVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"

#include <limits>

using namespace mlir;

namespace ondrix::ondsp {

LogicalResult verifyProductPolicy(Operation *op, Attribute numeric,
                                  std::optional<ProductAttr> product) {
  if (isa<FixedAttr>(numeric)) {
    if (!product)
      return op->emitOpError("fixed numeric policy requires a product attribute");
    if (failed(inferProductSemantics(op, cast<FixedAttr>(numeric), *product)))
      return failure();
    return success();
  }

  if (product)
    return op->emitOpError("floating-point numeric policy must not specify a product attribute");
  return success();
}

llvm::StringRef getFastPermissionAttrName() { return "ondsp.fast_used"; }

Value consumeFastPermission(Operation *op, FastPermission permission) {
  MLIRContext *context = op->getContext();
  StringRef spelling = permission == FastPermission::RebuildReductionTree ? "rebuild_reduction_tree"
                                                                          : "fuse_multiply_add";
  op->setAttr(getFastPermissionAttrName(), StringAttr::get(context, spelling));
  return op->getResult(0);
}

void summarizeFastPermissions(ModuleOp module) {
  SetVector<StringRef> spent;
  if (auto existing = module->getAttrOfType<ArrayAttr>(getFastPermissionAttrName()))
    for (Attribute entry : existing)
      spent.insert(cast<StringAttr>(entry).getValue());
  module.walk([&](Operation *op) {
    if (auto record = op->getAttrOfType<StringAttr>(getFastPermissionAttrName()))
      spent.insert(record.getValue());
  });
  if (spent.empty())
    return;
  SmallVector<StringRef> sorted(spent.begin(), spent.end());
  llvm::sort(sorted);
  SmallVector<Attribute> entries;
  for (StringRef name : sorted)
    entries.push_back(StringAttr::get(module.getContext(), name));
  module->setAttr(getFastPermissionAttrName(), ArrayAttr::get(module.getContext(), entries));
}

LogicalResult verifyExecutableFpFormat(Operation *op, FpAttr numeric, StringRef executable) {
  if (!numeric.getFormat().isF32())
    return op->emitOpError() << "executable " << executable
                             << " supports the f32 floating-point format";
  return success();
}

static LogicalResult verifyButterflyScale(Operation *op, ScaleAttr scale, unsigned rightShift,
                                          unsigned storageWidth, StringRef name) {
  if (scale.getPreShiftLeft() != 0 || scale.getPostShiftRight() != rightShift)
    return op->emitOpError() << name
                             << " requires pre_shift_left=0 and post_shift_right=" << rightShift;
  if (scale.getRounding() != RoundingMode::NearestEven)
    return op->emitOpError() << name << " requires nearest_even rounding";
  if (scale.getOverflow() != OverflowMode::Saturate)
    return op->emitOpError() << name << " requires saturating overflow";
  auto destination = dyn_cast<IntegerType>(scale.getSaturateTo());
  if (!destination || !destination.isSignless() || destination.getWidth() != storageWidth)
    return op->emitOpError() << name << " requires signless i" << storageWidth
                             << " destination storage";
  return success();
}

std::optional<PackedComplexProfile> getPackedComplexProfile(ComplexLayout layout) {
  switch (layout) {
  case ComplexLayout::PackedI16ImagHiRealLo:
    return PackedComplexProfile{16, 32};
  case ComplexLayout::PackedI32ImagHiRealLo:
    return PackedComplexProfile{32, 64};
  case ComplexLayout::Split:
  case ComplexLayout::Interleaved:
  case ComplexLayout::PackedI16RealHiImagLo:
    return std::nullopt;
  }
  return std::nullopt;
}

LogicalResult verifyPackedButterflyPolicy(Operation *op, CxLayoutAttr layout, Attribute numeric,
                                          ProductAttr product, ScaleAttr productScale,
                                          ScaleAttr outputScale) {
  std::optional<PackedComplexProfile> profile = getPackedComplexProfile(layout.getLayout());
  if (!profile)
    return op->emitOpError("executable butterfly requires packed_i16_imag_hi_real_lo or "
                           "packed_i32_imag_hi_real_lo layout");
  unsigned storageWidth = profile->storageWidth;
  auto fixed = dyn_cast<FixedAttr>(numeric);
  auto storage = fixed ? dyn_cast<IntegerType>(fixed.getStorage()) : IntegerType();
  if (!fixed || !storage || !storage.isSignless() || storage.getWidth() != storageWidth ||
      fixed.getFrac() != storageWidth - 1 || fixed.getSignedness() != Signedness::Signed)
    return op->emitOpError() << "packed butterfly requires signed Q" << (storageWidth - 1)
                             << " numeric semantics for this layout";
  if (!isFullProduct(product))
    return op->emitOpError("packed butterfly requires product = #ondsp.product<full>");
  // One product requantization per product term and one output scale per
  // stage: both boundaries are declared, never folded into each other.
  if (failed(
          verifyButterflyScale(op, productScale, storageWidth - 1, storageWidth, "product_scale")))
    return failure();
  return verifyButterflyScale(op, outputScale, 1, storageWidth, "output_scale");
}

FailureOr<ProductSemantics> inferProductSemantics(Operation *op, FixedAttr numeric,
                                                  ProductAttr product) {
  if (numeric.getSignedness() != Signedness::Signed)
    return op->emitOpError("fixed product semantics currently require a signed numeric policy");

  auto storage = cast<IntegerType>(numeric.getStorage());
  uint64_t storageWidth = storage.getWidth();
  uint64_t frac = numeric.getFrac();
  if (frac > std::numeric_limits<uint64_t>::max() / 2)
    return op->emitOpError("product fractional position is unrepresentable");

  uint64_t productFrac = frac * 2;
  switch (product.getSelection()) {
  case ProductSelection::Full:
    if (storageWidth > std::numeric_limits<unsigned>::max() / 2)
      return op->emitOpError("full product storage width is unrepresentable");
    if (storageWidth * 2 > std::numeric_limits<unsigned>::max() ||
        productFrac > std::numeric_limits<unsigned>::max())
      return op->emitOpError("product fractional position is unrepresentable");
    return ProductSemantics{static_cast<unsigned>(storageWidth * 2),
                            static_cast<unsigned>(productFrac), ProductSelection::Full};
  case ProductSelection::HighRaw:
    if (productFrac < storageWidth)
      return op->emitOpError("raw high product fractional position would be negative");
    productFrac -= storageWidth;
    if (storageWidth > std::numeric_limits<unsigned>::max() ||
        productFrac > std::numeric_limits<unsigned>::max())
      return op->emitOpError("product fractional position is unrepresentable");
    return ProductSemantics{static_cast<unsigned>(storageWidth), static_cast<unsigned>(productFrac),
                            ProductSelection::HighRaw};
  }
  return op->emitOpError("unsupported fixed product selection");
}

ReductionReassociationSafety classifyReductionReassociation(OverflowMode updateOverflow) {
  if (updateOverflow == OverflowMode::Wrap)
    return ReductionReassociationSafety::ExactModulo;
  return ReductionReassociationSafety::MustPreserveOrder;
}

TransformLegality TransformLegality::getAlgebraicIdentity() {
  return TransformLegality(TransformExactness::Exact, TransformJustification::AlgebraicIdentity);
}

TransformLegality TransformLegality::getFixedWidthModulo() {
  return TransformLegality(TransformExactness::Exact, TransformJustification::FixedWidthModulo);
}

TransformLegality TransformLegality::getIllegal() {
  return TransformLegality(TransformExactness::Illegal, TransformJustification::None);
}

TransformLegality classifyZeroProductElimination(Attribute numeric) {
  return isa<FixedAttr>(numeric) ? TransformLegality::getAlgebraicIdentity()
                                 : TransformLegality::getIllegal();
}

FailureOr<DistributivePairingSemantics> classifyDistributiveProductPairing(Operation *op,
                                                                           FixedAttr numeric,
                                                                           ProductAttr product,
                                                                           AccType accumulator) {
  FailureOr<ProductSemantics> productSemantics = inferProductSemantics(op, numeric, product);
  if (failed(productSemantics))
    return failure();

  bool exactBeforeAccumulatorOverflow = numeric.getSignedness() == Signedness::Signed &&
                                        productSemantics->selection == ProductSelection::Full &&
                                        accumulator.getSignedness() == numeric.getSignedness() &&
                                        accumulator.getFrac() == productSemantics->frac;
  TransformLegality legality = TransformLegality::getIllegal();
  if (exactBeforeAccumulatorOverflow && accumulator.getUpdateOverflow() == OverflowMode::Wrap)
    legality = TransformLegality::getFixedWidthModulo();
  return DistributivePairingSemantics{*productSemantics, legality, exactBeforeAccumulatorOverflow};
}

void appendAccumulatorCandidateTypes(Operation *op, SmallVectorImpl<Type> &types) {
  llvm::append_range(types, op->getOperandTypes());
  llvm::append_range(types, op->getResultTypes());
  if (auto function = dyn_cast<FunctionOpInterface>(op)) {
    llvm::append_range(types, function.getArgumentTypes());
    llvm::append_range(types, function.getResultTypes());
  }
  for (Region &region : op->getRegions())
    for (Block &block : region)
      llvm::append_range(types, block.getArgumentTypes());
}

AccType findRejectedAccumulator(Type type, llvm::function_ref<bool(AccType)> accepted) {
  AccType rejected;
  type.walk([&](AccType accumulator) {
    if (accepted(accumulator))
      return WalkResult::advance();
    rejected = accumulator;
    return WalkResult::interrupt();
  });
  return rejected;
}

bool isSingleLaneAccumulator(AccType accumulator) { return accumulator.getLanes() == 1; }

bool isSignedQ15(FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 16 && numeric.getFrac() == 15 &&
         numeric.getSignedness() == Signedness::Signed;
}

bool isSignedQ31(FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 32 && numeric.getFrac() == 31 &&
         numeric.getSignedness() == Signedness::Signed;
}

bool isSignedI40Frac30Accumulator(AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 40 &&
         accumulator.getFrac() == 30 && accumulator.getSignedness() == Signedness::Signed;
}

bool isSignedI64Frac62Accumulator(AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == 64 &&
         accumulator.getFrac() == 62 && accumulator.getSignedness() == Signedness::Signed;
}

bool isFullProduct(ProductAttr product) { return product.getSelection() == ProductSelection::Full; }

bool isRawHighProduct(ProductAttr product) {
  return product.getSelection() == ProductSelection::HighRaw;
}

} // namespace ondrix::ondsp
