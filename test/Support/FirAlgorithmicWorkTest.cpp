#include "ondrix/Analysis/FirAlgorithmicWork.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace {

mlir::FailureOr<ondrix::ConstantSequenceFacts> makeFacts(mlir::MLIRContext &context,
                                                         llvm::ArrayRef<int16_t> values) {
  auto type = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())},
                                          mlir::IntegerType::get(&context, 16));
  auto elements = mlir::DenseIntElementsAttr::get(type, values);
  return ondrix::analyzeConstantIntegerSequence(elements, values.size());
}

bool expectEqual(llvm::StringRef label, int64_t actual, int64_t expected) {
  if (actual == expected)
    return true;
  llvm::errs() << label << ": expected " << expected << ", got " << actual << '\n';
  return false;
}

bool testOrderedReduction() {
  auto estimate = ondrix::analysis::estimateOrderedFirWork(9);
  if (mlir::failed(estimate)) {
    llvm::errs() << "ordered reduction: analysis failed\n";
    return false;
  }

  bool ok = true;
  ok &= expectEqual("ordered taps", estimate->tapCount, 9);
  ok &= expectEqual("ordered input loads", estimate->scalarInputLoadExecutions, 9);
  ok &= expectEqual("ordered coefficient loads", estimate->scalarCoefficientLoadExecutions, 9);
  ok &= expectEqual("ordered products", estimate->scalarProductExecutions, 9);
  ok &= expectEqual("ordered updates", estimate->accumulatorUpdateExecutions, 9);
  if (mlir::succeeded(ondrix::analysis::estimateOrderedFirWork(-1))) {
    llvm::errs() << "ordered negative tap count: unexpectedly succeeded\n";
    ok = false;
  }
  return ok;
}

bool testConstantSpecialization() {
  mlir::MLIRContext context;
  auto symmetric = makeFacts(context, {0, 2, 3, 2, 0});
  auto evenSymmetric = makeFacts(context, {4, 0, 0, 4});
  auto allZero = makeFacts(context, {0, 0, 0});
  auto asymmetric = makeFacts(context, {1, 0, 2, 3});
  if (mlir::failed(symmetric) || mlir::failed(evenSymmetric) || mlir::failed(allZero) ||
      mlir::failed(asymmetric)) {
    llvm::errs() << "constant specialization: fact construction failed\n";
    return false;
  }

  ondrix::analysis::FirAlgorithmicWorkEstimate sparse =
      ondrix::analysis::estimateSparseConstantFirWork(*symmetric);
  auto paired = ondrix::analysis::estimateSymmetricConstantFirWorkAssumingLegal(*symmetric);
  auto evenPaired = ondrix::analysis::estimateSymmetricConstantFirWorkAssumingLegal(*evenSymmetric);
  ondrix::analysis::FirAlgorithmicWorkEstimate zeroSparse =
      ondrix::analysis::estimateSparseConstantFirWork(*allZero);
  if (mlir::failed(paired) || mlir::failed(evenPaired)) {
    llvm::errs() << "constant specialization: symmetric estimate failed\n";
    return false;
  }

  bool ok = true;
  ok &= expectEqual("sparse taps", sparse.tapCount, 5);
  ok &= expectEqual("sparse input loads", sparse.scalarInputLoadExecutions, 3);
  ok &= expectEqual("sparse coefficient uses", sparse.scalarConstantCoefficientUses, 3);
  ok &= expectEqual("sparse products", sparse.scalarProductExecutions, 3);
  ok &= expectEqual("sparse updates", sparse.accumulatorUpdateExecutions, 3);
  ok &= expectEqual("paired input loads", paired->scalarInputLoadExecutions, 3);
  ok &= expectEqual("paired coefficient uses", paired->scalarConstantCoefficientUses, 2);
  ok &= expectEqual("paired products", paired->scalarProductExecutions, 2);
  ok &= expectEqual("paired pre-adds", paired->widenedPreAddExecutions, 1);
  ok &= expectEqual("paired updates", paired->accumulatorUpdateExecutions, 2);
  ok &= expectEqual("even paired input loads", evenPaired->scalarInputLoadExecutions, 2);
  ok &= expectEqual("even paired products", evenPaired->scalarProductExecutions, 1);
  ok &= expectEqual("even paired pre-adds", evenPaired->widenedPreAddExecutions, 1);
  ok &= expectEqual("even paired updates", evenPaired->accumulatorUpdateExecutions, 1);
  ok &= expectEqual("all-zero sparse input loads", zeroSparse.scalarInputLoadExecutions, 0);
  ok &= expectEqual("all-zero sparse products", zeroSparse.scalarProductExecutions, 0);
  ok &= expectEqual("all-zero sparse updates", zeroSparse.accumulatorUpdateExecutions, 0);
  if (mlir::succeeded(
          ondrix::analysis::estimateSymmetricConstantFirWorkAssumingLegal(*asymmetric))) {
    llvm::errs() << "asymmetric estimate: unexpectedly succeeded\n";
    ok = false;
  }
  return ok;
}

bool testVectorCandidates() {
  auto ordered = ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
      19, 8, ondrix::analysis::FirVectorUpdateShape::OrderedLanes);
  auto horizontal = ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
      19, 8, ondrix::analysis::FirVectorUpdateShape::HorizontalChunks);
  auto shortReduction = ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
      7, 8, ondrix::analysis::FirVectorUpdateShape::HorizontalChunks);
  auto exactChunks = ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
      16, 8, ondrix::analysis::FirVectorUpdateShape::HorizontalChunks);
  if (mlir::failed(ordered) || mlir::failed(horizontal) || mlir::failed(shortReduction) ||
      mlir::failed(exactChunks)) {
    llvm::errs() << "Vector candidate: analysis failed\n";
    return false;
  }

  bool ok = true;
  ok &= expectEqual("ordered Vector chunks", ordered->vectorChunkExecutions, 2);
  ok &= expectEqual("ordered Vector tail", ordered->scalarTailElementCount, 3);
  ok &= expectEqual("ordered Vector input loads", ordered->vectorInputLoadExecutions, 2);
  ok &=
      expectEqual("ordered Vector coefficient loads", ordered->vectorCoefficientLoadExecutions, 2);
  ok &= expectEqual("ordered scalar input loads", ordered->scalarInputLoadExecutions, 3);
  ok &=
      expectEqual("ordered scalar coefficient loads", ordered->scalarCoefficientLoadExecutions, 3);
  ok &= expectEqual("ordered Vector products", ordered->vectorProductExecutions, 2);
  ok &= expectEqual("ordered scalar products", ordered->scalarProductExecutions, 3);
  ok &= expectEqual("ordered horizontal reductions", ordered->horizontalReductionExecutions, 0);
  ok &= expectEqual("ordered accumulator updates", ordered->accumulatorUpdateExecutions, 19);
  ok &= expectEqual("horizontal reductions", horizontal->horizontalReductionExecutions, 2);
  ok &= expectEqual("horizontal accumulator updates", horizontal->accumulatorUpdateExecutions, 5);
  ok &= expectEqual("short Vector chunks", shortReduction->vectorChunkExecutions, 0);
  ok &= expectEqual("short Vector tail", shortReduction->scalarTailElementCount, 7);
  ok &= expectEqual("short Vector updates", shortReduction->accumulatorUpdateExecutions, 7);
  ok &= expectEqual("exact Vector chunks", exactChunks->vectorChunkExecutions, 2);
  ok &= expectEqual("exact Vector tail", exactChunks->scalarTailElementCount, 0);
  ok &= expectEqual("exact Vector updates", exactChunks->accumulatorUpdateExecutions, 2);
  if (mlir::succeeded(ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
          -1, 8, ondrix::analysis::FirVectorUpdateShape::OrderedLanes))) {
    llvm::errs() << "negative Vector tap count: unexpectedly succeeded\n";
    ok = false;
  }
  if (mlir::succeeded(ondrix::analysis::estimateVectorFirWorkAssumingUpdateShapeLegal(
          8, 1, ondrix::analysis::FirVectorUpdateShape::HorizontalChunks))) {
    llvm::errs() << "invalid Vector width: unexpectedly succeeded\n";
    ok = false;
  }
  return ok;
}

} // namespace

int main() {
  if (!testOrderedReduction() || !testConstantSpecialization() || !testVectorCandidates()) {
    llvm::errs() << "FIR algorithmic work analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "FIR algorithmic work analysis: PASS\n";
  return 0;
}
