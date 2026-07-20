#ifndef ONDRIX_SUPPORT_FIRSTREAMRUNTIMESHAPE_H
#define ONDRIX_SUPPORT_FIRSTREAMRUNTIMESHAPE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace ondrix::ir {
class FirStreamOp;
}

namespace ondrix {

void emitFirStreamRuntimeShapeAssertions(ir::FirStreamOp op, mlir::Value inputLength,
                                         mlir::Value coefficientLength, mlir::Value stateLength,
                                         mlir::Value zero, mlir::Value one,
                                         mlir::OpBuilder &builder);

} // namespace ondrix

#endif // ONDRIX_SUPPORT_FIRSTREAMRUNTIMESHAPE_H
