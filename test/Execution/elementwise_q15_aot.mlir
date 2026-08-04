// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/elementwise_q15_aot.c %t.o -o %t
// RUN: %t

// Every unary member is swept over its whole 2^16 input domain, so the
// -32768 rail and every rounding tie are reached by construction. The
// harness also asserts that the four tie rules of a right shift by one give
// four different answers on 1, 3, -1, -3, which is the probe that separates
// them; without it the four kernels could be one contract in four spellings.

func.func @add_saturate(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.add %lhs, %rhs {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @add_wrap(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.add %lhs, %rhs {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @sub_saturate(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.sub %lhs, %rhs {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @mult_nearest_even(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @mult_ties_positive(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @mult_toward_negative(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @mult_toward_zero(%lhs: tensor<4096xi16>, %rhs: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.mult %lhs, %rhs {
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>, tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @abs_saturate(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.abs %input {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @abs_wrap(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.abs %input {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @negate_saturate(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.negate %input {
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @negate_wrap(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.negate %input {
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @offset_saturate(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.offset %input {
    bias = -12345 : i64,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_right_nearest_even(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_right_ties_positive(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_right_toward_negative(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_right_toward_zero(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = -1 : i64,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_left_saturate(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = 4 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @shift_left_wrap(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.shift %input {
    amount = 4 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}
