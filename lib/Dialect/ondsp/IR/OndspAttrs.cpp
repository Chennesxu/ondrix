#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

using namespace mlir;
using namespace ondrix::ondsp;

#include "ondrix/Dialect/ondsp/IR/OndspEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.cpp.inc"

void OndspDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.cpp.inc"
      >();
}

LogicalResult FixedAttr::verify(function_ref<InFlightDiagnostic()> emitError, Signedness signedness,
                                Type storage, unsigned frac) {
  (void)signedness;
  auto intType = storage.dyn_cast<IntegerType>();
  if (!intType)
    return emitError() << "fixed numeric storage must be an integer type";

  if (!intType.isSignless())
    return emitError() << "fixed numeric storage must use a signless integer type";

  unsigned width = intType.getWidth();
  if (width == 0)
    return emitError() << "fixed numeric storage must have non-zero width";

  if (frac > width)
    return emitError() << "fixed numeric frac must not exceed storage width";

  return success();
}

LogicalResult FpAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type format,
                             FpContractMode contract) {
  (void)contract;
  if (!format.isa<FloatType>())
    return emitError() << "floating-point numeric format must be a floating-point type";
  return success();
}

LogicalResult ScaleAttr::verify(function_ref<InFlightDiagnostic()> emitError, unsigned preShiftLeft,
                                unsigned postShiftRight, RoundingMode rounding,
                                OverflowMode overflow, Type saturateTo) {
  (void)rounding;
  (void)overflow;

  auto destinationType = saturateTo.dyn_cast<IntegerType>();
  if (!destinationType || !destinationType.isSignless())
    return emitError() << "scale saturate_to must be a signless integer type";

  constexpr unsigned kMaxStaticShift = 63;
  if (preShiftLeft > kMaxStaticShift)
    return emitError() << "scale pre_shift_left must be at most " << kMaxStaticShift;

  if (postShiftRight > kMaxStaticShift)
    return emitError() << "scale post_shift_right must be at most " << kMaxStaticShift;

  return success();
}
