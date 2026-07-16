#include "ondrix/Conversion/Utils/ConversionLegality.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace ondrix::conversion {

bool containsMatchingType(Type type, TypePredicate predicate) {
  return type
      .walk([&](Type nested) {
        return predicate(nested) ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

bool containsMatchingType(TypeRange types, TypePredicate predicate) {
  return llvm::any_of(types, [&](Type type) { return containsMatchingType(type, predicate); });
}

bool containsMatchingType(Attribute attribute, TypePredicate predicate) {
  return attribute
      .walk([&](Type nested) {
        return predicate(nested) ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

bool containsMatchingAttribute(Attribute attribute, AttributePredicate predicate) {
  return attribute
      .walk([&](Attribute nested) {
        return predicate(nested) ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

bool hasLegalConvertedTypesAndAttributes(Operation *op, TypeConverter &typeConverter,
                                         TypePredicate rejectedType,
                                         AttributePredicate rejectedAttribute) {
  if (!typeConverter.isLegal(op) || containsMatchingType(op->getOperandTypes(), rejectedType) ||
      containsMatchingType(op->getResultTypes(), rejectedType))
    return false;

  for (NamedAttribute namedAttribute : op->getAttrs()) {
    Attribute value = namedAttribute.getValue();
    if (containsMatchingType(value, rejectedType) ||
        containsMatchingAttribute(value, rejectedAttribute))
      return false;
  }

  for (Region &region : op->getRegions()) {
    if (!typeConverter.isLegal(&region))
      return false;
    for (Block &block : region)
      if (containsMatchingType(block.getArgumentTypes(), rejectedType))
        return false;
  }
  return true;
}

LogicalResult verifySCFWhileTypeConversionSafety(Operation *root, TypePredicate convertedType) {
  WalkResult result = root->walk([&](scf::WhileOp op) {
    if (op->getAttrs().empty())
      return WalkResult::advance();
    if (!containsMatchingType(op->getOperandTypes(), convertedType) &&
        !containsMatchingType(op->getResultTypes(), convertedType))
      return WalkResult::advance();
    op.emitOpError("attributes on scf.while carrying converted source types are unsupported by "
                   "the LLVM 17 structural type conversion");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

} // namespace ondrix::conversion
