#ifndef ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H
#define ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <utility>

namespace ondrix {

/// Properties inferred from an immutable integer sequence.
class ConstantSequenceFacts {
public:
  llvm::ArrayRef<llvm::APInt> getValues() const { return values; }
  int64_t getElementCount() const { return elementCount; }
  bool hasZero() const { return containsZero; }
  bool isSymmetric() const { return symmetric; }

private:
  friend mlir::FailureOr<ConstantSequenceFacts>
  analyzeConstantIntegerSequence(mlir::DenseIntElementsAttr elements, int64_t maxElements);

  llvm::SmallVector<llvm::APInt> values;
  int64_t elementCount = 0;
  bool containsZero = false;
  bool symmetric = true;
};

/// Integer sequence facts bound to one memref value proven to cover an entire
/// constant rank-one global. Only the corresponding analysis entry point can
/// construct this evidence.
class ConstantIntegerMemRefFacts final {
public:
  mlir::Value getSource() const { return source; }
  mlir::Value getRoot() const { return root; }
  const ConstantSequenceFacts &getSequence() const { return sequence; }

private:
  friend mlir::FailureOr<ConstantIntegerMemRefFacts>
  analyzeConstantIntegerMemRef(mlir::Value value, int64_t maxElements);

  ConstantIntegerMemRefFacts(mlir::Value source, mlir::Value root, ConstantSequenceFacts sequence)
      : source(source), root(root), sequence(std::move(sequence)) {}

  mlir::Value source;
  mlir::Value root;
  ConstantSequenceFacts sequence;
};

/// Infers integer sequence properties without materializing more than
/// `maxElements` values.
mlir::FailureOr<ConstantSequenceFacts>
analyzeConstantIntegerSequence(mlir::DenseIntElementsAttr elements, int64_t maxElements);

/// Resolves a constant rank-one integer `memref.global` through either a
/// direct `memref.get_global`, proven full-range unit-stride subviews, or
/// metadata-only `memref.cast` operations. Partial, strided, rank-changing,
/// memory-space-changing, and unprovably dynamic views fail closed.
mlir::FailureOr<ConstantIntegerMemRefFacts> analyzeConstantIntegerMemRef(mlir::Value value,
                                                                         int64_t maxElements);

} // namespace ondrix

#endif // ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H
