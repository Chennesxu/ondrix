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

std::unique_ptr<mlir::Pass> createDecomposeOndrixFirStreamPass();

std::unique_ptr<mlir::Pass> createEvaluateOndrixFirDesignPass();
std::unique_ptr<mlir::Pass> createFuseOndrixElementwiseChainsPass();
std::unique_ptr<mlir::Pass> createMergeOndrixGainCascadesPass();

std::unique_ptr<mlir::Pass> createFuseOndrixGainIntoFirPass();
std::unique_ptr<mlir::Pass>
createFuseOndrixGainIntoFirPass(const FuseOndrixGainIntoFirOptions &options);

std::unique_ptr<mlir::Pass> createForwardOndrixInsertExtractPass();
std::unique_ptr<mlir::Pass> createConvertOndrixStaticResultsToOutParamsPass();
std::unique_ptr<mlir::Pass> createDeclareOndrixArgumentsReadOnlyPass();
std::unique_ptr<mlir::Pass> createForwardOndrixResultBuffersPass();
std::unique_ptr<mlir::Pass>
createForwardOndrixResultBuffersPass(const ForwardOndrixResultBuffersOptions &options);
std::unique_ptr<mlir::Pass> createApplyOndrixLlvmArgumentAttributesPass();

/// Function attribute naming the expanded LLVM argument positions that
/// `apply-ondrix-llvm-argument-attributes` marks `llvm.noalias`.
constexpr llvm::StringLiteral kNoAliasPointerArgumentsAttr = "ondrix.noalias_pointer_args";
std::unique_ptr<mlir::Pass> createDeclareOndrixCEntryPointsPass();
std::unique_ptr<mlir::Pass> createEmitOndrixCEntryPointsPass();
std::unique_ptr<mlir::Pass>
createEmitOndrixCEntryPointsPass(const EmitOndrixCEntryPointsOptions &options);
std::unique_ptr<mlir::Pass> createUnfuseOndspSharedFpProductsPass();
std::unique_ptr<mlir::Pass> createRelaxOndspUnreachableSaturationPass();
std::unique_ptr<mlir::Pass> createScalarizeOndspFixedReduceMacPass();
std::unique_ptr<mlir::Pass> createScalarizeOndspCertifiedConstantReducePass();
std::unique_ptr<mlir::Pass> createUnrollOndspFixedMacLoopsPass();
std::unique_ptr<mlir::Pass> createLowerOndspAssumptionsPass();
std::unique_ptr<mlir::Pass> createUnrollOndspFpOrderedReducePass();
std::unique_ptr<mlir::Pass> createWidenOndspExactAccumulatorsPass();

#define GEN_PASS_REGISTRATION
#include "ondrix/Transforms/Passes.h.inc"

} // namespace ondrix

#endif // ONDRIX_TRANSFORMS_PASSES_H
