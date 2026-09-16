// RUN: ondrix-opt %s --scalarize-ondsp-certified-constant-reduce --scalarize-ondsp-fixed-reduce-mac --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.certified.mlir
// RUN: ondrix-translate %t.certified.mlir --mlir-to-llvmir > %t.certified.ll
// RUN: FileCheck %s --check-prefix=CERTIFIED < %t.certified.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.certified.ll -o %t.certified.o
// RUN: cc %S/Inputs/certified_constant_reduce_aot.c %t.certified.o -o %t.certified.bin
// RUN: %t.certified.bin > %t.certified.txt
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.declared.mlir
// RUN: ondrix-translate %t.declared.mlir --mlir-to-llvmir > %t.declared.ll
// RUN: FileCheck %s --check-prefix=DECLARED < %t.declared.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.declared.ll -o %t.declared.o
// RUN: cc %S/Inputs/certified_constant_reduce_aot.c %t.declared.o -o %t.declared.bin
// RUN: %t.declared.bin > %t.declared.txt
// RUN: diff %t.certified.txt %t.declared.txt

// The certified expansion never clamps at the i40 rails; the declared
// expansion clamps there after every product, and the two must agree.
// CERTIFIED-NOT: 549755813887
// CERTIFIED: mul i32
// CERTIFIED-NOT: 549755813887
// DECLARED: -549755813888

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @pairs : memref<8xi16> = dense<[32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768]>
memref.global "private" constant @small : memref<8xi16> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
memref.global "private" constant @long : memref<70xi16> = dense<3>

func.func @dot_pairs(%x: memref<8xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %c = memref.get_global @pairs : memref<8xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}

func.func @dot_reversed(%x: memref<8xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %c = memref.get_global @small : memref<8xi16>
  %r = memref.subview %c[7] [8] [-1] : memref<8xi16> to memref<8xi16, strided<[-1], offset: 7>>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %r {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16, strided<[-1], offset: 7>>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}

func.func @dot_long(%x: memref<70xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %c = memref.get_global @long : memref<70xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<70xi16>, memref<70xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}
