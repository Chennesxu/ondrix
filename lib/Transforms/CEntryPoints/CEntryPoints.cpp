#include "ondrix/Transforms/CEntryPoints.h"
#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

namespace ondrix {
#define GEN_PASS_DEF_DECLAREONDRIXCENTRYPOINTS
#define GEN_PASS_DEF_EMITONDRIXCENTRYPOINTS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// The element and result types C can spell: fixed-width integers and binary32/64.
bool isCScalar(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.isSignless() && llvm::is_contained({8u, 16u, 32u, 64u}, integer.getWidth());
  return type.isF32() || type.isF64();
}

/// A buffer the entry can pass as one pointer: a window of any extent, or a
/// static shape of any rank (its sizes and strides are constants).
bool isEntryBuffer(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() >= 1 && memref.getLayout().isIdentity() &&
         !memref.getMemorySpace() && isCScalar(memref.getElementType()) &&
         (memref.getRank() == 1 || memref.hasStaticShape());
}

bool hasScalarResult(FunctionType type) {
  return type.getNumResults() == 1 && isCScalar(type.getResult(0));
}

/// Static memref results become trailing out-parameters before the entry is built.
bool isSupportedEntrySignature(FunctionType type) {
  return llvm::all_of(type.getInputs(), isEntryBuffer) &&
         (hasScalarResult(type) || llvm::all_of(type.getResults(), [](Type result) {
            return isEntryBuffer(result) && cast<MemRefType>(result).hasStaticShape();
          }));
}

/// The buffers in the expanded kernel's parameter order: the inputs, then the
/// results that became out-parameters.
SmallVector<MemRefType> getEntryBuffers(FunctionType type) {
  SmallVector<MemRefType> buffers;
  for (Type input : type.getInputs())
    buffers.push_back(cast<MemRefType>(input));
  if (!hasScalarResult(type))
    for (Type result : type.getResults())
      buffers.push_back(cast<MemRefType>(result));
  return buffers;
}

std::string getCEntryName(StringRef kernel) { return (ondrix::kCEntryPrefix + kernel).str(); }

/// What the record carries: the bufferized signature, one name per buffer, and
/// per buffer the extent group it shares one length with (-1 static).
struct EntryRecord {
  FunctionType signature;
  SmallVector<std::string> names;
  SmallVector<int64_t> groups;
};

/// Groups are the compact ids 0..k-1, exactly on the dynamic extents.
bool hasWellFormedGroups(const EntryRecord &record) {
  int64_t next = 0;
  for (auto [buffer, group] : llvm::zip_equal(getEntryBuffers(record.signature), record.groups)) {
    if (buffer.hasStaticShape() != (group < 0) || group > next)
      return false;
    if (group == next)
      ++next;
  }
  return true;
}

FailureOr<EntryRecord> readRecord(Operation *op) {
  auto dictionary = op->getAttrOfType<DictionaryAttr>(ondrix::kCEntryAttr);
  auto signature =
      dictionary ? dyn_cast_or_null<TypeAttr>(dictionary.get("signature")) : TypeAttr();
  auto names = dictionary ? dyn_cast_or_null<ArrayAttr>(dictionary.get("names")) : ArrayAttr();
  auto groups = dictionary ? dyn_cast_or_null<DenseI64ArrayAttr>(dictionary.get("groups"))
                           : DenseI64ArrayAttr();
  auto type = signature ? dyn_cast<FunctionType>(signature.getValue()) : FunctionType();
  auto malformed = [&] { return op->emitOpError("carries a malformed C entry record"); };
  if (!type || !isSupportedEntrySignature(type) || !names || !groups)
    return malformed();
  int64_t numBuffers = getEntryBuffers(type).size();
  if (static_cast<int64_t>(names.size()) != numBuffers || groups.size() != numBuffers ||
      !llvm::all_of(names, [](Attribute name) { return isa<StringAttr>(name); }))
    return malformed();
  EntryRecord record{type, {}, {}};
  for (Attribute name : names)
    record.names.push_back(cast<StringAttr>(name).str());
  record.groups.assign(groups.asArrayRef().begin(), groups.asArrayRef().end());
  if (!hasWellFormedGroups(record))
    return malformed();
  return record;
}

DictionaryAttr writeRecord(MLIRContext *context, const EntryRecord &record) {
  Builder builder(context);
  SmallVector<Attribute> names;
  for (const std::string &name : record.names)
    names.push_back(builder.getStringAttr(name));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("groups", builder.getDenseI64ArrayAttr(record.groups)),
       builder.getNamedAttr("names", builder.getArrayAttr(names)),
       builder.getNamedAttr("signature", TypeAttr::get(record.signature))});
}

/// The entry-block argument a reduction operand reads, through memref casts.
BlockArgument getWindowArgument(func::FuncOp function, Value operand) {
  while (auto cast = operand.getDefiningOp<memref::CastOp>())
    operand = cast.getSource();
  auto argument = dyn_cast<BlockArgument>(operand);
  if (!argument || argument.getOwner() != &function.getBody().front() ||
      !cast<MemRefType>(argument.getType()).isDynamicDim(0))
    return nullptr;
  return argument;
}

/// Extent groups: `ondsp.reduce_mac` reads two equal-length windows, so two
/// dynamic parameters it pairs share one length; every other dynamic extent
/// is its own group, numbered by first parameter. Results are static.
SmallVector<int64_t> deriveExtentGroups(func::FuncOp function, unsigned numBuffers) {
  unsigned numInputs = function.getNumArguments();
  SmallVector<int64_t> parent(numInputs);
  for (unsigned index = 0; index < numInputs; ++index)
    parent[index] = index;
  auto find = [&](int64_t index) {
    while (parent[index] != index)
      index = parent[index];
    return index;
  };
  function.walk([&](ondrix::ondsp::ReduceMacOp reduce) {
    BlockArgument lhs = getWindowArgument(function, reduce.getLhs());
    BlockArgument rhs = getWindowArgument(function, reduce.getRhs());
    if (lhs && rhs)
      parent[std::max(find(lhs.getArgNumber()), find(rhs.getArgNumber()))] =
          std::min(find(lhs.getArgNumber()), find(rhs.getArgNumber()));
  });
  SmallVector<int64_t> groups(numBuffers, -1);
  int64_t next = 0;
  for (unsigned index = 0; index < numInputs; ++index) {
    if (cast<MemRefType>(function.getArgument(index).getType()).hasStaticShape())
      continue;
    int64_t root = find(index);
    groups[index] = groups[root] >= 0 ? groups[root] : next++;
  }
  return groups;
}

int64_t getGroupCount(ArrayRef<int64_t> groups) {
  return groups.empty() ? 0 : *std::max_element(groups.begin(), groups.end()) + 1;
}

/// The wrapper's parameter list: one pointer per buffer, and each group's length
/// right after the group's last buffer.
struct EntryLayout {
  SmallVector<int64_t> pointerPosition;
  SmallVector<int64_t> lengthPosition;
  unsigned numParams = 0;
};

EntryLayout getEntryLayout(ArrayRef<int64_t> groups) {
  EntryLayout layout;
  int64_t numGroups = getGroupCount(groups);
  SmallVector<int64_t> lastMember(numGroups, -1);
  for (auto [index, group] : llvm::enumerate(groups))
    if (group >= 0)
      lastMember[group] = index;
  layout.lengthPosition.assign(numGroups, -1);
  for (auto [index, group] : llvm::enumerate(groups)) {
    layout.pointerPosition.push_back(layout.numParams++);
    if (group >= 0 && lastMember[group] == static_cast<int64_t>(index))
      layout.lengthPosition[group] = layout.numParams++;
  }
  return layout;
}

class DeclareOndrixCEntryPointsPass final
    : public ondrix::impl::DeclareOndrixCEntryPointsBase<DeclareOndrixCEntryPointsPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal() || !function.isPublic() ||
          !function->hasAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName()))
        continue;
      EntryRecord record{function.getFunctionType(), {}, {}};
      if (!isSupportedEntrySignature(record.signature))
        continue;
      for (BlockArgument argument : function.getArguments()) {
        auto name = dyn_cast<NameLoc>(argument.getLoc());
        record.names.push_back(name ? name.getName().str()
                                    : llvm::formatv("a{0}", argument.getArgNumber()).str());
      }
      unsigned numBuffers = getEntryBuffers(record.signature).size();
      unsigned numOutputs = numBuffers - function.getNumArguments();
      for (unsigned index = 0; index < numOutputs; ++index) {
        std::string name = numOutputs == 1 ? "output" : llvm::formatv("output{0}", index).str();
        while (llvm::is_contained(record.names, name))
          name += '_';
        record.names.push_back(name);
      }
      record.groups = deriveExtentGroups(function, numBuffers);
      std::string entry = getCEntryName(function.getName());
      if (symbols.lookup(entry)) {
        function.emitOpError() << "C entry point '" << entry
                               << "' collides with an existing symbol";
        return signalPassFailure();
      }
      function->setAttr(ondrix::kCEntryAttr, writeRecord(&getContext(), record));
    }
  }
};

/// The message-then-abort shape `convert-cf-to-llvm` gives a failed assertion.
void emitAbort(ModuleOp module, OpBuilder &builder, Location loc, StringRef message) {
  MLIRContext *context = module.getContext();
  auto pointerType = LLVM::LLVMPointerType::get(context);
  auto voidType = LLVM::LLVMVoidType::get(context);
  LLVM::LLVMFuncOp puts = LLVM::lookupOrCreateFn(module, "puts", pointerType, voidType);
  LLVM::LLVMFuncOp abortFn = LLVM::lookupOrCreateFn(module, "abort", {}, voidType);
  SymbolTable symbols(module);
  std::string name;
  for (unsigned counter = 0;; ++counter) {
    name = llvm::formatv("ondrix_c_entry_msg_{0}", counter);
    if (!symbols.lookup(name))
      break;
  }
  std::string terminated = message.str();
  terminated.push_back('\0');
  Value text = LLVM::createGlobalString(loc, builder, name, terminated, LLVM::Linkage::Private,
                                        /*useOpaquePointers=*/true);
  builder.create<LLVM::CallOp>(loc, puts, text);
  builder.create<LLVM::CallOp>(loc, abortFn, ValueRange());
}

LogicalResult emitEntry(ModuleOp module, LLVM::LLVMFuncOp kernel, bool checked) {
  FailureOr<EntryRecord> record = readRecord(kernel);
  if (failed(record))
    return failure();
  FunctionType signature = record->signature;
  ArrayRef<int64_t> groups = record->groups;
  SmallVector<MemRefType> buffers = getEntryBuffers(signature);

  // Each buffer expanded to (allocated, aligned, offset, sizes..., strides...);
  // the index width is whatever the conversion chose.
  LLVM::LLVMFunctionType kernelType = kernel.getFunctionType();
  auto mismatch = [&] {
    return kernel.emitOpError("recorded C entry signature does not match the expanded function");
  };
  Type returnType = hasScalarResult(signature) ? signature.getResult(0)
                                               : LLVM::LLVMVoidType::get(module.getContext());
  unsigned numFields = 0;
  for (MemRefType buffer : buffers)
    numFields += 3 + 2 * buffer.getRank();
  if (kernelType.getNumParams() != numFields || kernelType.getReturnType() != returnType)
    return mismatch();
  IntegerType indexType;
  unsigned field = 0;
  for (MemRefType buffer : buffers) {
    Type pointer = kernelType.getParamType(field);
    auto offset = dyn_cast<IntegerType>(kernelType.getParamType(field + 2));
    if (!isa<LLVM::LLVMPointerType>(pointer) || pointer != kernelType.getParamType(field + 1) ||
        !offset || (indexType && indexType != offset))
      return mismatch();
    for (unsigned extent = 3; extent < 3 + 2 * buffer.getRank(); ++extent)
      if (kernelType.getParamType(field + extent) != offset)
        return mismatch();
    indexType = offset;
    field += 3 + 2 * buffer.getRank();
  }

  EntryLayout layout = getEntryLayout(groups);
  auto pointerType = LLVM::LLVMPointerType::get(module.getContext());
  SmallVector<Type> params(layout.numParams);
  for (auto [index, position] : llvm::enumerate(layout.pointerPosition))
    params[position] = pointerType;
  for (int64_t position : layout.lengthPosition)
    params[position] = indexType;

  Location loc = kernel.getLoc();
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  auto entry = builder.create<LLVM::LLVMFuncOp>(loc, getCEntryName(kernel.getName()),
                                                LLVM::LLVMFunctionType::get(returnType, params));
  Block *body = entry.addEntryBlock();
  builder.setInsertionPointToStart(body);

  DenseMap<int64_t, Value> constants;
  auto constant = [&](int64_t value) {
    Value &cached = constants[value];
    if (!cached)
      cached = builder.create<LLVM::ConstantOp>(loc, indexType, value);
    return cached;
  };
  auto elementBytes = [](MemRefType buffer) -> int64_t {
    return llvm::divideCeil(buffer.getElementType().getIntOrFloatBitWidth(), 8);
  };
  auto allOf = [&](Value all, Value term) {
    return all ? builder.create<LLVM::AndOp>(loc, all, term).getResult() : term;
  };
  // Each check refuses with its own message; a passing call falls through.
  auto refuseUnless = [&](Value ok, StringRef message) {
    if (!ok)
      return;
    Block *proceed = entry.addBlock();
    Block *refuse = entry.addBlock();
    builder.create<LLVM::CondBrOp>(loc, ok, proceed, refuse);
    builder.setInsertionPointToStart(refuse);
    emitAbort(module, builder, loc, llvm::formatv("{0}: {1}", entry.getName(), message).str());
    builder.create<LLVM::UnreachableOp>(loc);
    builder.setInsertionPointToStart(proceed);
  };

  // A length is read back as a signed index and scaled to bytes, so the widest
  // member of its group bounds it.
  SmallVector<int64_t> groupBytes(layout.lengthPosition.size(), 1);
  for (auto [index, buffer] : llvm::enumerate(buffers))
    if (groups[index] >= 0)
      groupBytes[groups[index]] = std::max(groupBytes[groups[index]], elementBytes(buffer));
  Value fits;
  for (auto [group, position] : llvm::enumerate(layout.lengthPosition)) {
    APInt bound = APInt::getSignedMaxValue(indexType.getWidth()).udiv(groupBytes[group]);
    Value limit =
        builder.create<LLVM::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, bound));
    fits = allOf(fits, builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule,
                                                    body->getArgument(position), limit));
  }
  refuseUnless(fits, "a length exceeds the addressable range");

  if (checked) {
    SmallVector<Value> begins, ends;
    Value nonNull;
    for (auto [index, buffer] : llvm::enumerate(buffers)) {
      Value pointer = body->getArgument(layout.pointerPosition[index]);
      Value begin = builder.create<LLVM::PtrToIntOp>(loc, indexType, pointer);
      Value bytes = buffer.hasStaticShape()
                        ? constant(buffer.getNumElements() * elementBytes(buffer))
                        : builder
                              .create<LLVM::MulOp>(
                                  loc, body->getArgument(layout.lengthPosition[groups[index]]),
                                  constant(elementBytes(buffer)))
                              .getResult();
      begins.push_back(begin);
      ends.push_back(builder.create<LLVM::AddOp>(loc, begin, bytes));
      Value empty = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, bytes, constant(0));
      Value present =
          builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, begin, constant(0));
      nonNull = allOf(nonNull, builder.create<LLVM::OrOp>(loc, empty, present));
    }
    // Half-open byte ranges are disjoint when the later start is at or past
    // the earlier end; an empty range overlaps nothing.
    Value disjoint;
    for (unsigned first = 0; first < begins.size(); ++first)
      for (unsigned second = first + 1; second < begins.size(); ++second) {
        Value start = builder.create<LLVM::UMaxOp>(loc, begins[first], begins[second]);
        Value stop = builder.create<LLVM::UMinOp>(loc, ends[first], ends[second]);
        disjoint = allOf(disjoint,
                         builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, start, stop));
      }
    refuseUnless(nonNull, "a null buffer has elements");
    refuseUnless(disjoint, "two buffers overlap");
  }

  SmallVector<Value> operands;
  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    Value pointer = body->getArgument(layout.pointerPosition[index]);
    operands.append({pointer, pointer, constant(0)});
    if (!buffer.hasStaticShape()) {
      operands.append({body->getArgument(layout.lengthPosition[groups[index]]), constant(1)});
      continue;
    }
    for (int64_t size : buffer.getShape())
      operands.push_back(constant(size));
    int64_t stride = 1;
    SmallVector<Value> strides(buffer.getRank());
    for (int64_t dim = buffer.getRank() - 1; dim >= 0; --dim) {
      strides[dim] = constant(stride);
      stride *= buffer.getDimSize(dim);
    }
    operands.append(strides);
  }
  auto result = builder.create<LLVM::CallOp>(loc, kernel, operands);
  builder.create<LLVM::ReturnOp>(loc, result.getResults());

  entry->setAttr(ondrix::kCEntryAttr, kernel->getAttr(ondrix::kCEntryAttr));
  kernel->removeAttr(ondrix::kCEntryAttr);
  return success();
}

class EmitOndrixCEntryPointsPass final
    : public ondrix::impl::EmitOndrixCEntryPointsBase<EmitOndrixCEntryPointsPass> {
public:
  using ondrix::impl::EmitOndrixCEntryPointsBase<
      EmitOndrixCEntryPointsPass>::EmitOndrixCEntryPointsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto function : module.getOps<LLVM::LLVMFuncOp>()) {
      if (!function->hasAttr(ondrix::kCEntryAttr))
        continue;
      // `convert-func-to-llvm` copies the kernel's attributes onto the
      // descriptor-pointer wrapper it emits; that copy is not a record.
      if (function.getName().startswith("_mlir_ciface_")) {
        function->removeAttr(ondrix::kCEntryAttr);
        continue;
      }
      kernels.push_back(function);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(emitEntry(module, kernel, checked)))
        return signalPassFailure();
  }
};

StringRef getCTypeName(Type type) {
  if (type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  switch (cast<IntegerType>(type).getWidth()) {
  case 8:
    return "int8_t";
  case 16:
    return "int16_t";
  case 32:
    return "int32_t";
  default:
    return "int64_t";
  }
}

} // namespace

std::unique_ptr<Pass> ondrix::createDeclareOndrixCEntryPointsPass() {
  return std::make_unique<DeclareOndrixCEntryPointsPass>();
}

std::unique_ptr<Pass> ondrix::createEmitOndrixCEntryPointsPass() {
  return std::make_unique<EmitOndrixCEntryPointsPass>();
}

std::unique_ptr<Pass>
ondrix::createEmitOndrixCEntryPointsPass(const EmitOndrixCEntryPointsOptions &options) {
  return std::make_unique<EmitOndrixCEntryPointsPass>(options);
}

LogicalResult ondrix::printOndrixCEntryHeader(ModuleOp module, raw_ostream &os) {
  os << "// Plain-pointer entry points generated by Ondrix. Buffers are contiguous,\n"
        "// row-major, element-aligned and valid for the call, and no two may overlap;\n"
        "// an input is read only, an output receives the whole result. A length counts\n"
        "// elements and must fit the signed index range in bytes, or the call prints a\n"
        "// message and aborts before touching memory. Built with --checked-entries, a\n"
        "// null buffer with elements or an overlap is refused the same way.\n"
        "#include <stdint.h>\n\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
  for (auto entry : module.getOps<LLVM::LLVMFuncOp>()) {
    if (!entry->hasAttr(kCEntryAttr) || !entry.getName().startswith(kCEntryPrefix))
      continue;
    FailureOr<EntryRecord> record = readRecord(entry);
    if (failed(record))
      return failure();
    ArrayRef<int64_t> groups = record->groups;
    EntryLayout layout = getEntryLayout(groups);
    // A record still on a kernel has the expanded shape, not the entry's.
    if (entry.getNumArguments() != layout.numParams)
      continue;
    int64_t numGroups = getGroupCount(groups);
    SmallVector<std::string> fields(layout.numParams);
    for (auto [index, buffer] : llvm::enumerate(getEntryBuffers(record->signature))) {
      bool input = index < record->signature.getNumInputs();
      fields[layout.pointerPosition[index]] =
          llvm::formatv("{0}{1} *{2}", input ? "const " : "", getCTypeName(buffer.getElementType()),
                        record->names[index]);
      int64_t group = groups[index];
      if (group < 0 || !fields[layout.lengthPosition[group]].empty())
        continue;
      unsigned width =
          cast<IntegerType>(entry.getArgument(layout.lengthPosition[group]).getType()).getWidth();
      fields[layout.lengthPosition[group]] =
          llvm::formatv("uint{0}_t {1}", width,
                        numGroups == 1 ? std::string("length") : record->names[index] + "_length");
    }
    os << (hasScalarResult(record->signature) ? getCTypeName(record->signature.getResult(0))
                                              : StringRef("void"))
       << ' ' << entry.getName() << '('
       << (fields.empty() ? std::string("void") : llvm::join(fields, ", ")) << ");\n";
  }
  os << "\n#ifdef __cplusplus\n}\n#endif\n";
  return success();
}
