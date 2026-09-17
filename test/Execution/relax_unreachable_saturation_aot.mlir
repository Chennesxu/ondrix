// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --scalarize-ondsp-certified-constant-reduce --relax-ondsp-unreachable-saturation --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.relaxed.mlir
// RUN: ondrix-translate %t.relaxed.mlir --mlir-to-llvmir > %t.relaxed.ll
// RUN: FileCheck %s --check-prefix=RELAXED < %t.relaxed.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.relaxed.ll -o %t.relaxed.o
// RUN: cc %S/Inputs/relax_unreachable_saturation_aot.c %t.relaxed.o -o %t.relaxed.bin
// RUN: %t.relaxed.bin > %t.relaxed.txt
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --scalarize-ondsp-certified-constant-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.declared.mlir
// RUN: ondrix-translate %t.declared.mlir --mlir-to-llvmir > %t.declared.ll
// RUN: FileCheck %s --check-prefix=DECLARED < %t.declared.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.declared.ll -o %t.declared.o
// RUN: cc %S/Inputs/relax_unreachable_saturation_aot.c %t.declared.o -o %t.declared.bin
// RUN: %t.declared.bin > %t.declared.txt
// RUN: diff %t.relaxed.txt %t.declared.txt

// The declared object tests every reading against the i16 rails and selects;
// the relaxed object has no comparison at all, and the two must agree.
// DECLARED: icmp eq
// DECLARED: select
// RELAXED-NOT: icmp
// RELAXED-NOT: select

func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16> attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}
