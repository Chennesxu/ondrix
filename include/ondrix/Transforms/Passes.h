#ifndef ONDRIX_TRANSFORMS_PASSES_H
#define ONDRIX_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace ondrix {

#define GEN_PASS_DECL
#include "ondrix/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createSpecializeOndrixConstantFirPass();
std::unique_ptr<mlir::Pass>
createSpecializeOndrixConstantFirPass(const SpecializeOndrixConstantFirOptions &options);

std::unique_ptr<mlir::Pass> createTileOndrixFirFilterPass();
std::unique_ptr<mlir::Pass>
createTileOndrixFirFilterPass(const TileOndrixFirFilterOptions &options);

#define GEN_PASS_REGISTRATION
#include "ondrix/Transforms/Passes.h.inc"

} // namespace ondrix

#endif // ONDRIX_TRANSFORMS_PASSES_H
