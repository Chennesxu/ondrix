#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"

#include <limits>
#include <tuple>

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
llvm::StringRef getFastSelectionAttrName() { return "ondsp.fast_selection"; }
llvm::StringRef getFastSourceSiteAttrName() { return "ondsp.fast_source_site"; }

namespace {

StringRef spellPermission(FastPermission permission) {
  return permission == FastPermission::RebuildReductionTree ? "rebuild_reduction_tree"
                                                            : "fuse_multiply_add";
}

/// The ordinal of `op` among operations of its own name in the enclosing
/// symbol. Computed only when no earlier pass stamped an id; a collision it
/// could still produce is reported by the summary rather than tolerated.
std::string computeSourceSiteId(Operation *op) {
  auto symbol = op->getParentOfType<FunctionOpInterface>();
  StringRef scope = symbol ? SymbolTable::getSymbolName(symbol).getValue() : "<no-symbol>";
  int64_t ordinal = 0;
  if (symbol) {
    StringRef name = op->getName().getStringRef();
    symbol->walk([&](Operation *other) {
      if (other == op)
        return WalkResult::interrupt();
      if (other->getName().getStringRef() == name)
        ++ordinal;
      return WalkResult::advance();
    });
  }
  return (scope + "/" + op->getName().getStringRef() + "#" + Twine(ordinal)).str();
}

/// Field order is the printed order, so the dictionary is byte-stable.
DictionaryAttr buildSelectionAttr(MLIRContext *context, const FastSelectionPlan &plan,
                                  ArrayRef<StringRef> used) {
  SmallVector<Attribute> permissions;
  for (StringRef name : used)
    permissions.push_back(StringAttr::get(context, name));
  SmallVector<NamedAttribute> fields{
      {StringAttr::get(context, "instance_domain"), StringAttr::get(context, plan.instanceDomain)},
      {StringAttr::get(context, "mechanism"), StringAttr::get(context, plan.mechanism)},
      {StringAttr::get(context, "route_role"), StringAttr::get(context, plan.routeRole)},
      {StringAttr::get(context, "source_operation"),
       StringAttr::get(context, plan.sourceOperation)},
      {StringAttr::get(context, "source_site_id"), StringAttr::get(context, plan.sourceSiteId)},
      {StringAttr::get(context, "used_permissions"), ArrayAttr::get(context, permissions)},
      {StringAttr::get(context, "when"), StringAttr::get(context, plan.condition)}};
  return DictionaryAttr::get(context, fields);
}

} // namespace

FastSelectionPlan planFastSelection(Operation *sourceOp, StringRef routeRole,
                                    StringRef instanceDomain, StringRef mechanism,
                                    StringRef condition) {
  FastSelectionPlan plan;
  if (auto stamped = sourceOp->getAttrOfType<DictionaryAttr>(getFastSourceSiteAttrName()))
    plan.sourceSiteId = stamped.getAs<StringAttr>("source_site_id").getValue().str();
  else
    plan.sourceSiteId = computeSourceSiteId(sourceOp);
  plan.sourceOperation = sourceOp->getName().getStringRef();
  plan.routeRole = routeRole;
  plan.instanceDomain = instanceDomain;
  plan.mechanism = mechanism;
  plan.condition = condition;
  return plan;
}

void stampFastSourceSite(Operation *op, const FastSelectionPlan &plan) {
  MLIRContext *context = op->getContext();
  SmallVector<NamedAttribute> fields{
      {StringAttr::get(context, "instance_domain"), StringAttr::get(context, plan.instanceDomain)},
      {StringAttr::get(context, "route_role"), StringAttr::get(context, plan.routeRole)},
      {StringAttr::get(context, "source_operation"),
       StringAttr::get(context, plan.sourceOperation)},
      {StringAttr::get(context, "source_site_id"), StringAttr::get(context, plan.sourceSiteId)}};
  op->setAttr(getFastSourceSiteAttrName(), DictionaryAttr::get(context, fields));
}

bool declaresFastPermissions(Attribute numeric) {
  auto fp = dyn_cast_or_null<FpAttr>(numeric);
  return fp && fp.getContract() == FpContractMode::Fast;
}

FastSelectionPlan continueFastSelection(Operation *op, StringRef mechanism, StringRef condition) {
  FastSelectionPlan plan;
  auto stamped = op->getAttrOfType<DictionaryAttr>(getFastSourceSiteAttrName());
  if (stamped) {
    plan.sourceSiteId = stamped.getAs<StringAttr>("source_site_id").getValue().str();
    plan.sourceOperation = stamped.getAs<StringAttr>("source_operation").getValue();
    plan.routeRole = stamped.getAs<StringAttr>("route_role").getValue();
    plan.instanceDomain = stamped.getAs<StringAttr>("instance_domain").getValue();
  } else {
    plan.sourceSiteId = computeSourceSiteId(op);
    plan.sourceOperation = op->getName().getStringRef();
    plan.routeRole = "whole";
    plan.instanceDomain = "0 <= i < N";
  }
  plan.mechanism = mechanism;
  plan.condition = condition;
  return plan;
}

Value consumeFastPermission(Operation *event, FastPermission permission,
                            const FastSelectionPlan &plan) {
  MLIRContext *context = event->getContext();
  SetVector<StringRef> used;
  // A site may spend both, and the previous single string silently overwrote.
  if (auto existing = event->getAttrOfType<DictionaryAttr>(getFastSelectionAttrName()))
    if (auto prior = existing.getAs<ArrayAttr>("used_permissions"))
      for (Attribute entry : prior)
        used.insert(cast<StringAttr>(entry).getValue());
  used.insert(spellPermission(permission));
  SmallVector<StringRef> sorted(used.begin(), used.end());
  llvm::sort(sorted);
  event->setAttr(getFastSelectionAttrName(), buildSelectionAttr(context, plan, sorted));
  return event->getResult(0);
}

LogicalResult summarizeFastPermissions(ModuleOp module) {
  MLIRContext *context = module.getContext();
  // The route role is part of the identity, not a detail of it: one source
  // operation owns several roles, and each may generate a conditional schedule
  // per branch. Several events under one case spend different permissions - the
  // lane body spends F while the cross-lane fold spends R - so the case's set
  // is their union and disagreement is only about the rest.
  using RecordKey = std::tuple<StringRef, StringRef, StringRef>;
  struct Case {
    DictionaryAttr shape;
    SetVector<StringRef> used;
  };
  llvm::MapVector<RecordKey, Case> cases;
  SetVector<StringRef> spent;

  WalkResult walk = module.walk([&](Operation *op) {
    auto record = op->getAttrOfType<DictionaryAttr>(getFastSelectionAttrName());
    if (!record)
      return WalkResult::advance();
    auto site = record.getAs<StringAttr>("source_site_id");
    auto role = record.getAs<StringAttr>("route_role");
    auto when = record.getAs<StringAttr>("when");
    auto used = record.getAs<ArrayAttr>("used_permissions");
    if (!site || !role || !when || !used) {
      op->emitError("fast selection record is missing a required field");
      return WalkResult::interrupt();
    }
    RecordKey key{site.getValue(), role.getValue(), when.getValue()};
    auto [it, inserted] = cases.insert({key, Case{record, {}}});
    if (!inserted &&
        it->second.shape.getAs<StringAttr>("mechanism") != record.getAs<StringAttr>("mechanism")) {
      op->emitError("one route and condition reports two mechanisms: ")
          << site.getValue() << '/' << role.getValue();
      return WalkResult::interrupt();
    }
    for (Attribute entry : used) {
      StringRef name = cast<StringAttr>(entry).getValue();
      it->second.used.insert(name);
      spent.insert(name);
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  // Replaced when there is something to replace it with, never accumulated: a
  // union outlives the decision it described. Not cleared when nothing is
  // found, because a record can legitimately outlive the events it came from -
  // vector.reduction's custom form drops its attribute dictionary, so a second
  // invocation over printed IR sees no events and must not erase what it cannot
  // recompute. Input arriving with a record of its own is refused at entry
  // instead, by refuseForgedFastRecords.
  if (cases.empty())
    return success();
  module->removeAttr(getFastSelectionAttrName());
  module->removeAttr(getFastPermissionAttrName());

  SmallVector<DictionaryAttr> records;
  for (auto &entry : cases) {
    SmallVector<StringRef> used(entry.second.used.begin(), entry.second.used.end());
    llvm::sort(used);
    SmallVector<Attribute> permissions;
    for (StringRef name : used)
      permissions.push_back(StringAttr::get(context, name));
    SmallVector<NamedAttribute> fields;
    for (NamedAttribute field : entry.second.shape)
      fields.push_back(field.getName() == "used_permissions"
                           ? NamedAttribute(field.getName(), ArrayAttr::get(context, permissions))
                           : field);
    records.push_back(DictionaryAttr::get(context, fields));
  }
  llvm::sort(records, [](DictionaryAttr lhs, DictionaryAttr rhs) {
    auto key = [](DictionaryAttr attr) {
      return std::make_tuple(attr.getAs<StringAttr>("source_site_id").getValue(),
                             attr.getAs<StringAttr>("route_role").getValue(),
                             attr.getAs<StringAttr>("when").getValue());
    };
    return key(lhs) < key(rhs);
  });
  SmallVector<Attribute> entries(records.begin(), records.end());
  module->setAttr(getFastSelectionAttrName(), ArrayAttr::get(context, entries));

  SmallVector<StringRef> sortedSpent(spent.begin(), spent.end());
  llvm::sort(sortedSpent);
  SmallVector<Attribute> permissions;
  for (StringRef name : sortedSpent)
    permissions.push_back(StringAttr::get(context, name));
  module->setAttr(getFastPermissionAttrName(), ArrayAttr::get(context, permissions));
  return success();
}

LogicalResult refuseForgedFastRecords(ModuleOp module) {
  StringRef names[] = {getFastPermissionAttrName(), getFastSelectionAttrName(),
                       getFastSourceSiteAttrName()};
  WalkResult walk = module.walk([&](Operation *op) {
    for (StringRef name : names)
      if (op->hasAttr(name)) {
        op->emitError("'")
            << name
            << "' is a compiler-owned audit attribute; input must not carry a decision record";
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
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
