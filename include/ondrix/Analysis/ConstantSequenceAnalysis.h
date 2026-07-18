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

/// Integer sequence facts bound to one direct constant memref global value.
/// Only the corresponding analysis entry point can construct this evidence.
class DirectConstantIntegerMemRefFacts final {
public:
  mlir::Value getSource() const { return source; }
  const ConstantSequenceFacts &getSequence() const { return sequence; }

private:
  friend mlir::FailureOr<DirectConstantIntegerMemRefFacts>
  analyzeDirectConstantIntegerMemRefGlobal(mlir::Value value, int64_t maxElements);

  DirectConstantIntegerMemRefFacts(mlir::Value source, ConstantSequenceFacts sequence)
      : source(source), sequence(std::move(sequence)) {}

  mlir::Value source;
  ConstantSequenceFacts sequence;
};

/// Infers integer sequence properties without materializing more than
/// `maxElements` values.
mlir::FailureOr<ConstantSequenceFacts>
analyzeConstantIntegerSequence(mlir::DenseIntElementsAttr elements, int64_t maxElements);

/// Resolves a direct `memref.get_global` of a constant rank-one integer global
/// and infers its sequence properties. Views and casts are intentionally not
/// followed until their immutability and indexing semantics are modeled.
mlir::FailureOr<DirectConstantIntegerMemRefFacts>
analyzeDirectConstantIntegerMemRefGlobal(mlir::Value value, int64_t maxElements);

} // namespace ondrix

#endif // ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H
