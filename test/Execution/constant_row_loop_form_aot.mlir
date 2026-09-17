// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128" > %t.line.mlir
// RUN: ondrix-translate %t.line.mlir --mlir-to-llvmir > %t.line.ll
// RUN: FileCheck %s --check-prefix=LINE < %t.line.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.line.ll -o %t.line.o
// RUN: cc %S/Inputs/constant_row_loop_form_aot.c %t.line.o -o %t.line.bin
// RUN: %t.line.bin > %t.line.txt
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128 max-straight-line-coefficients=32" > %t.loop.mlir
// RUN: ondrix-translate %t.loop.mlir --mlir-to-llvmir > %t.loop.ll
// RUN: FileCheck %s --check-prefix=LOOP < %t.loop.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loop.ll -o %t.loop.o
// RUN: cc %S/Inputs/constant_row_loop_form_aot.c %t.loop.o -o %t.loop.bin
// RUN: %t.loop.bin > %t.loop.txt
// RUN: diff %t.line.txt %t.loop.txt

// The two forms differ only in where the coefficients live: the straight-line
// object materializes them, the loop object reads one column table.
// LINE-NOT: __ondrix_row_table_
// LOOP: @__ondrix_row_table_0 = private constant

func.func @dct32_q15(%input: tensor<32xi16>) -> tensor<32xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 9>
  } : (tensor<32xi16>) -> tensor<32xi16>
  return %result : tensor<32xi16>
}
