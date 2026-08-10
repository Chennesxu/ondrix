#ifndef ONDRIX_LIB_CONVERSION_ONDRIXTOONDSP_ONDRIXTOONDSPCOMMON_H
#define ONDRIX_LIB_CONVERSION_ONDRIXTOONDSP_ONDRIXTOONDSPCOMMON_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace ondrix::conversion {

/// One declared floating-point accumulator update under the call site's
/// contract mode; the fast mode spends its fuse permission here.
mlir::Value createFpAccumulatorUpdate(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                                      mlir::Value accumulator, ondsp::FpAttr numeric,
                                      mlir::OpBuilder &builder);

/// Contract-invariant product and sum sites (nothing to fuse, declared order).
mlir::Value createFpMultiply(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                             mlir::OpBuilder &builder);
mlir::Value createFpAdd(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                        mlir::OpBuilder &builder);

mlir::Value createEmptyTensor(mlir::Location loc, mlir::RankedTensorType type,
                              mlir::Value dynamicLength, mlir::OpBuilder &builder);

ondsp::ScaleAttr getNearestEvenSaturatingShift(mlir::MLIRContext *context, unsigned shift);

/// Pattern registration per algorithm family; together these are exactly the
/// convert-ondrix-to-ondsp pattern set.
void populateOndrixFirFamilyLoweringPatterns(mlir::RewritePatternSet &patterns,
                                             bool slidingWindowReuse);
void populateOndrixStatefulLoweringPatterns(mlir::RewritePatternSet &patterns);
void populateOndrixSpectralLoweringPatterns(mlir::RewritePatternSet &patterns,
                                            bool vectorizeStaticCfft, bool fftLoops);
void populateOndrixElementwiseLoweringPatterns(mlir::RewritePatternSet &patterns);

} // namespace ondrix::conversion

#endif // ONDRIX_LIB_CONVERSION_ONDRIXTOONDSP_ONDRIXTOONDSPCOMMON_H
