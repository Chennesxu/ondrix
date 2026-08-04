// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.plain.mlir
// RUN: ondrix-translate %t.plain.mlir --mlir-to-llvmir > %t.plain.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.plain.ll -o %t.plain.o
// RUN: ondrix-opt %S/Inputs/legality_elementwise_fusion_q15_fused.mlir --fuse-ondrix-elementwise-chains --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.fused.mlir
// RUN: ondrix-translate %t.fused.mlir --mlir-to-llvmir > %t.fused.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fused.ll -o %t.fused.o
// RUN: cc %S/Inputs/legality_elementwise_fusion_q15_aot.c %t.plain.o %t.fused.o -o %t
// RUN: %t

// The certificate decides the rewrite from a model of the contract; this
// gate checks the other half over the same whole 2^16 domain — that the
// objects agree where it certified, and that they DIVERGE where it refused,
// so a refusal is a witnessed fact rather than a claim.

#q15 = #ondsp.fixed<signed, storage = i16, frac = 15>

func.func @negate_wrap_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @shift_directed_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @offset_wrap_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.offset %a {bias = 1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.offset %0 {bias = -1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @abs_of_negate_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.abs %0 {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

// The refused pairs, and beside each the single operation the rewrite would
// have produced had the certificate accepted it.

func.func @negate_saturate_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @negate_saturate_identity(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  return %a : tensor<4096xi16>
}

func.func @shift_nearest_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @shift_nearest_single(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.shift %a {amount = -5 : i64, numeric = #q15,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %0 : tensor<4096xi16>
}

func.func @offset_saturate_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.offset %a {bias = 1000 : i64, numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.offset %0 {bias = -1000 : i64, numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @offset_saturate_identity(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  return %a : tensor<4096xi16>
}
