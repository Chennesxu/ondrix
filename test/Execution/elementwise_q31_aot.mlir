// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/elementwise_q31_aot.c %t.o -o %t
// RUN: %t

// The same eighteen members at the wider width, where a 2^32 sweep is not
// available and the carrier is what has to be proven instead. The harness
// leads with the cases that separate an i64 body from an i32 one --
// mult(INT32_MIN, INT32_MIN), abs(INT32_MIN), shift_left(INT32_MAX, 4) --
// and each of the three is wrong, not merely imprecise, at the narrower
// carrier.

func.func @add_saturate(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.add %lhs, %rhs {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @add_wrap(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.add %lhs, %rhs {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @sub_saturate(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.sub %lhs, %rhs {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @mult_nearest_even(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @mult_ties_positive(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @mult_toward_negative(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @mult_toward_zero(%lhs: tensor<4096xi32>, %rhs: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @abs_saturate(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.abs %input {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @abs_wrap(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.abs %input {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @negate_saturate(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.negate %input {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @negate_wrap(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.negate %input {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @offset_saturate(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.offset %input {
    bias = -1234567890 : i64,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_right_nearest_even(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_right_ties_positive(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_right_toward_negative(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_right_toward_zero(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_left_saturate(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = 4 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @shift_left_wrap(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = 4 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}
