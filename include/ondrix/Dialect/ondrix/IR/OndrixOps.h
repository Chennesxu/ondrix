#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_H
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "ondrix/Dialect/ondrix/IR/OndrixAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h.inc"

namespace ondrix {
namespace ir {

/// The right shift `ondrix.rms` applies to each input before squaring, so that
/// the sum of `2^m` squares of a signed Q1.(storageWidth-1) sample stays exact
/// in i64. Zero at every admitted Q15 extent and nonzero at every Q31 one; the
/// verifier and the lowering must agree, so both read it here.
unsigned getRmsInputPreShift(unsigned storageWidth, int64_t extent);

/// The right shift a fixed reduction applies to each product before it joins
/// the sum, so that `terms` products of signed Q1.(storageWidth-1) operands
/// stay exact in i64. Zero at every admitted Q15 shape and at `terms == 1`.
/// `ondrix.matmul` reads it with K and `ondrix.lms` with the weight count;
/// verifiers, lowerings and the binding must agree, so all read it here.
unsigned getReductionProductShift(unsigned storageWidth, int64_t terms);

} // namespace ir
} // namespace ondrix

#endif
