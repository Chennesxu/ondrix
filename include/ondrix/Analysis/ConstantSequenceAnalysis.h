#ifndef ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H
#define ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

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

/// Infers integer sequence properties without materializing more than
/// `maxElements` values.
mlir::FailureOr<ConstantSequenceFacts>
analyzeConstantIntegerSequence(mlir::DenseIntElementsAttr elements, int64_t maxElements);

} // namespace ondrix

#endif // ONDRIX_ANALYSIS_CONSTANTSEQUENCEANALYSIS_H
