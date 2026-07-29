// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/trig_q15_aot.c %t.o -o %t
// RUN: %t

// EXHAUSTIVE gate: the phase domain is exactly the 65536 i16 values, so
// the harness sweeps every phase through both compiled functions in
// sixteen 4096-wide batches against an independent reference that embeds
// the mpmath-derived 256-entry table and the same interpolation contract
// in explicit floor-division form. Directed goldens pin the axes
// (sine(0) = 0, sine(quarter turn) = 32767 saturated, sine(half turn)
// = 0, sine(three quarters) = -32768) and the quarter-turn identity
// cosine(u) == sine(u + 16384) is checked across the whole domain.

func.func @sine4096_q15(%phase: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @cosine4096_q15(%phase: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}
