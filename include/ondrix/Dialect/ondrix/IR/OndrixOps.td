#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ondrix/Dialect/ondrix/IR/OndrixDialect.td"

def Ondrix_FirOp : Ondrix_Op<"fir", [Pure]> {
  let summary = "High-level FIR kernel intent";
  let arguments = (ins AnyType:$input, AnyType:$coeffs, AnyAttr:$numeric);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_DotOp : Ondrix_Op<"dot", [Pure]> {
  let summary = "High-level dot/reduce kernel intent";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyAttr:$numeric);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_ButterflyOp : Ondrix_Op<"butterfly", [Pure]> {
  let summary = "High-level complex radix-2 butterfly intent";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$twiddle,
                   AnyAttr:$layout, AnyAttr:$numeric);
  let results = (outs AnyType:$out0, AnyType:$out1);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_QuantizeOp : Ondrix_Op<"quantize", [Pure]> {
  let summary = "High-level numeric conversion intent";
  let arguments = (ins AnyType:$input, AnyAttr:$src, AnyAttr:$dst);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif
