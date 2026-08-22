#ifndef ONDRIX_SUPPORT_FPACCUMULATORUPDATE_H
#define ONDRIX_SUPPORT_FPACCUMULATORUPDATE_H

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

namespace ondrix {

/// One declared floating-point accumulator update under the call site's
/// contract mode; the fast mode spends its fuse permission here. Shared by
/// the ondrix-to-ondsp lowerings and the dialect's bufferization interface.
inline mlir::Value createFpAccumulatorUpdate(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                                             mlir::Value accumulator, ondsp::FpAttr numeric,
                                             mlir::OpBuilder &builder) {
  switch (numeric.getContract()) {
  case ondsp::FpContractMode::Off: {
    mlir::Value product = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::AddFOp>(loc, accumulator, product);
  }
  case ondsp::FpContractMode::Fma:
    return builder.create<mlir::math::FmaOp>(loc, lhs, rhs, accumulator);
  case ondsp::FpContractMode::Fast:
    // fast admits both members here; selecting the fused one spends F.
    return ondsp::consumeFastPermission(
        builder.create<mlir::math::FmaOp>(loc, lhs, rhs, accumulator),
        ondsp::FastPermission::FuseMultiplyAdd);
  }
  llvm_unreachable("unknown floating-point contract mode");
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_FPACCUMULATORUPDATE_H
