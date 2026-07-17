#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

#include "llvm/Support/raw_ostream.h"

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

} // namespace

int main() {
  if (!testSequenceProperties() || !testAnalysisLimit()) {
    llvm::errs() << "constant sequence analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "constant sequence analysis: PASS\n";
  return 0;
}
