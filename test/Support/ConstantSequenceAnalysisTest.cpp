#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace {

mlir::DenseIntElementsAttr makeSequence(mlir::MLIRContext &context,
                                        llvm::ArrayRef<int16_t> values) {
  auto type = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())},
                                          mlir::IntegerType::get(&context, 16));
  return mlir::DenseIntElementsAttr::get(type, values);
}

bool testSequenceProperties() {
  mlir::MLIRContext context;
  auto symmetric =
      ondrix::analyzeConstantIntegerSequence(makeSequence(context, {1, 2, 0, 2, 1}), 5);
  auto asymmetric =
      ondrix::analyzeConstantIntegerSequence(makeSequence(context, {1, 0, -2, 0, 3}), 5);
  if (mlir::failed(symmetric) || mlir::failed(asymmetric))
    return false;

  return symmetric->getElementCount() == 5 && symmetric->hasZero() && symmetric->isSymmetric() &&
         asymmetric->hasZero() && !asymmetric->isSymmetric();
}

bool testAnalysisLimit() {
  mlir::MLIRContext context;
  auto sequence = makeSequence(context, {1, 2, 3, 4, 5});
  return mlir::failed(ondrix::analyzeConstantIntegerSequence(sequence, 4)) &&
         mlir::succeeded(ondrix::analyzeConstantIntegerSequence(sequence, 5)) &&
         mlir::failed(ondrix::analyzeConstantIntegerSequence(sequence, -1));
}

bool testMemRefProvenance() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlir::Location loc = mlir::UnknownLoc::get(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(&context);
  auto i16 = builder.getI16Type();
  auto memrefType = mlir::MemRefType::get({5}, i16);
  auto initializer = makeSequence(context, {1, 2, 3, 4, 5});
  builder.setInsertionPointToStart(module->getBody());
  builder.create<mlir::memref::GlobalOp>(loc, "coefficients", builder.getStringAttr("private"),
                                         memrefType, initializer, true, mlir::IntegerAttr());
  builder.setInsertionPointToEnd(module->getBody());
  mlir::Value direct = builder.create<mlir::memref::GetGlobalOp>(loc, memrefType, "coefficients");
  mlir::Value dynamicSize = builder.create<mlir::memref::DimOp>(loc, direct, 0);
  mlir::Value full = builder.create<mlir::memref::SubViewOp>(
      loc, direct, mlir::OpFoldResult(builder.getIndexAttr(0)), mlir::OpFoldResult(dynamicSize),
      mlir::OpFoldResult(builder.getIndexAttr(1)));
  mlir::Value partial = builder.create<mlir::memref::SubViewOp>(
      loc, direct, mlir::OpFoldResult(builder.getIndexAttr(1)),
      mlir::OpFoldResult(builder.getIndexAttr(4)), mlir::OpFoldResult(builder.getIndexAttr(1)));
  mlir::Value strided = builder.create<mlir::memref::SubViewOp>(
      loc, direct, mlir::OpFoldResult(builder.getIndexAttr(0)),
      mlir::OpFoldResult(builder.getIndexAttr(3)), mlir::OpFoldResult(builder.getIndexAttr(2)));
  auto dynamicType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i16);
  mlir::Value cast = builder.create<mlir::memref::CastOp>(loc, dynamicType, full);

  auto directFacts = ondrix::analyzeConstantIntegerMemRef(direct, 5);
  auto fullFacts = ondrix::analyzeConstantIntegerMemRef(full, 5);
  auto castFacts = ondrix::analyzeConstantIntegerMemRef(cast, 5);
  return mlir::succeeded(directFacts) && mlir::succeeded(fullFacts) && mlir::succeeded(castFacts) &&
         directFacts->getSource() == direct && directFacts->getRoot() == direct &&
         fullFacts->getSource() == full && fullFacts->getRoot() == direct &&
         castFacts->getSource() == cast && castFacts->getRoot() == direct &&
         fullFacts->getSequence().getValues() == directFacts->getSequence().getValues() &&
         mlir::failed(ondrix::analyzeConstantIntegerMemRef(partial, 5)) &&
         mlir::failed(ondrix::analyzeConstantIntegerMemRef(strided, 5));
}

} // namespace

int main() {
  if (!testSequenceProperties()) {
    llvm::errs() << "constant sequence properties: FAIL\n";
    return 1;
  }
  if (!testAnalysisLimit()) {
    llvm::errs() << "constant sequence analysis limit: FAIL\n";
    return 1;
  }
  if (!testMemRefProvenance()) {
    llvm::errs() << "constant memref provenance: FAIL\n";
    return 1;
  }
  llvm::outs() << "constant sequence analysis: PASS\n";
  return 0;
}
