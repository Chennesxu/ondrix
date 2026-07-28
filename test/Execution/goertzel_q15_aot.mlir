// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/goertzel_q15_aot.c %t.o -o %t -lm
// RUN: %t

// Quantized-state Goertzel against an independent per-step reference
// whose coefficients embed mpmath-derived constants (not the compiler's
// cosine). Three profiles: an in-band bin of a power-of-two length, a
// bin of a NON-power-of-two length (no radix structure to fall back on),
// and the DC bin whose coefficient saturates to 32767 by declared
// convention. Corpus: an in-bin tone (large energy), an out-of-bin tone,
// full-scale rails driving the state saturation, and deterministic
// random trials — every energy must match the reference bit-exactly.

func.func @goertzel64_5(%input: tensor<64xi16>) -> tensor<1xi64>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

func.func @goertzel100_13(%input: tensor<100xi16>) -> tensor<1xi64>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 13 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<100xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

func.func @goertzel16_0(%input: tensor<16xi16>) -> tensor<1xi64>
    attributes {llvm.emit_c_interface} {
  %energy = ondrix.goertzel %input {
    bin = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
