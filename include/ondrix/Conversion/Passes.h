#ifndef ONDRIX_CONVERSION_PASSES_H
#define ONDRIX_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace ondrix {

#define GEN_PASS_DECL
#include "ondrix/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertOndrixToOndspPass();
std::unique_ptr<mlir::Pass> createConvertOrtumCoreToOndspEmulationPass();
std::unique_ptr<mlir::Pass> createConvertOndspFixedToScalarPass();
std::unique_ptr<mlir::Pass> createConvertOndspToOrtumCorePass();
std::unique_ptr<mlir::Pass> createLowerOndspF32ReduceToScalarPass();
std::unique_ptr<mlir::Pass> createNormalizeOndspFixedVectorReducePass();
std::unique_ptr<mlir::Pass> createParallelizeOndspFixedWrapVectorReducePass();
std::unique_ptr<mlir::Pass> createVectorizeOndspFixedMemRefReducePass();
std::unique_ptr<mlir::Pass>
createVectorizeOndspFixedMemRefReducePass(const VectorizeOndspFixedMemRefReduceOptions &options);

#define GEN_PASS_REGISTRATION
#include "ondrix/Conversion/Passes.h.inc"

} // namespace ondrix

#endif // ONDRIX_CONVERSION_PASSES_H
