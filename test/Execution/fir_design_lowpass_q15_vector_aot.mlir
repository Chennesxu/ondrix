// RUN: ondrix-opt %s --evaluate-ondrix-fir-design > %t.evaluated.mlir
// RUN: FileCheck %s --check-prefix=EVALUATED < %t.evaluated.mlir
// RUN: ondrix-opt %t.evaluated.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize > %t.bufferized.mlir
// RUN: ondrix-opt %t.bufferized.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64 proof-trace-output=%t.proof.json" > %t.proven.mlir
// RUN: FileCheck %s --check-prefix=PROVEN < %t.proven.mlir
// RUN: ondrix-opt %t.bufferized.mlir --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.proof.json max-elements=64" > /dev/null
// RUN: ondrix-opt %t.proven.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_design_lowpass_q15_vector_aot.c %t.o -o %t
// RUN: %t

// One design-to-execution program: the windowed-sinc lowpass profile is
// designed and quantized at compile time, the constant table feeds the
// symmetric no-overflow reassociation proof, the proven reduction runs as
// fixed-width Vector code, and the object is checked against an
// independent C reference that embeds the mpmath-derived golden taps.

// EVALUATED-LABEL: func.func @fir_design_lowpass_q15
// EVALUATED-NOT: ondrix.fir_design_windowed_sinc
// EVALUATED: arith.constant
// EVALUATED-SAME: response = "lowpass"
// EVALUATED-SAME: dense<[0, -747, 0, 9025, 16384, 9025, 0, -747, 0]> : tensor<9xi16>

// PROVEN-LABEL: func.func @fir_design_lowpass_q15
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN-NOT: = ondsp.reduce_mac

func.func @fir_design_lowpass_q15(%input: tensor<64xi16>) -> tensor<56xi16>
    attributes {llvm.emit_c_interface} {
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  %init = tensor.empty() : tensor<56xi16>
  %output = ondrix.fir_filter %input, %coefficients, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>, tensor<9xi16>, tensor<56xi16>) -> tensor<56xi16>
  return %output : tensor<56xi16>
}
