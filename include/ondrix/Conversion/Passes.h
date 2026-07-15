#ifndef ONDRIX_CONVERSION_PASSES_H
#define ONDRIX_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace ondrix {

std::unique_ptr<mlir::Pass> createConvertOndrixToOndspPass();
std::unique_ptr<mlir::Pass> createConvertOndspQ15ToScalarPass();
std::unique_ptr<mlir::Pass> createConvertOndspToOrtumCorePass();
std::unique_ptr<mlir::Pass> createLowerOndspF32ReduceToScalarPass();

#define GEN_PASS_DECL
#include "ondrix/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ondrix/Conversion/Passes.h.inc"

} // namespace ondrix

#endif // ONDRIX_CONVERSION_PASSES_H
