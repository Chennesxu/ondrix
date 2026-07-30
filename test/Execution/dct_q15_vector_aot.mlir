// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize > %t.bufferized.mlir
// RUN: ondrix-opt %t.bufferized.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=8 max-elements=64 proof-trace-output=%t.proof.json" > %t.proven.mlir
// RUN: FileCheck %s --check-prefix=PROVEN --implicit-check-not=ondsp.reduce_mac < %t.proven.mlir
// RUN: ondrix-opt %t.bufferized.mlir --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.proof.json max-elements=64" > /dev/null
// RUN: ondrix-opt %t.proven.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/dct_q15_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// The bufferization consumer of ondrix.dct reaches the Vector path through the
// OTHER legality route. matmul and rms reduce over runtime values, so they
// declare wrapping accumulators and reassociate as the exact-modulo class with
// no proof at all. A DCT row is a compile-time constant table, so this
// consumer declares a SATURATING i40 accumulator and the horizontal form is
// authorized only by the constant-coefficient prefix-range proof: every
// chunked prefix of a row is bounded by sum_n |c[k][n]| * 32768, at most
// 64 * 32767 * 32768 < 2^39, so no reassociated partial sum can saturate and
// the fold equals the exact i64 contract sum. The emitted proof trace is
// replayed against the original bufferized IR, and the object is checked
// against an independent C reference that embeds mpmath-derived tables.

// Every row must be proof-authorized: no memref-form reduction may survive.
// PROVEN-LABEL: func.func @dct8_q15_vector
// PROVEN: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// PROVEN-LABEL: func.func @dct32_q15_vector
// PROVEN: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// PROVEN-LABEL: func.func @dct64_q15_vector
// PROVEN: vector.reduction <add>, {{.*}} : vector<8xi64> into i64

// No ondsp operation may survive the pipeline.
// VECTOR-NOT: ondsp.

// The labels are anchored to line start so they cannot match the
// `_mlir_ciface_...` interface thunk emitted for the same kernel.
// AVX2-LABEL: {{^}}dct8_q15_vector:
// AVX2: vpmulld
// AVX2-LABEL: {{^}}dct32_q15_vector:
// AVX2: vpmulld
// AVX2-LABEL: {{^}}dct64_q15_vector:
// AVX2: vpmulld

// N = 8 is exactly one width-8 chunk per row: the reduction reassociates
// completely and no ordered scalar tail remains.
func.func @dct8_q15_vector(%input: tensor<8xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// N = 32 chains four width-8 chunks per row, so the proof has to cover the
// intermediate reassociated prefixes and not only the final sum.
func.func @dct32_q15_vector(%input: tensor<32xi16>) -> tensor<32xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 9>
  } : (tensor<32xi16>) -> tensor<32xi16>
  return %result : tensor<32xi16>
}

// N = 64 is the largest admitted extent: the maximal eight-chunk schedule,
// the largest coefficient table, and the tightest prefix bound the proof
// ever certifies — the DC row realizes sum |c| * 32768 = 2^36 - 2^21, the
// worst case admitted against the i40 rail of 2^39.
func.func @dct64_q15_vector(%input: tensor<64xi16>) -> tensor<64xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 8>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}
