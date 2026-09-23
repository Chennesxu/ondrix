// RUN: ondrix-opt %s --convert-ondsp-lane-split-to-ortumcore --convert-ondsp-to-ortumcore > %t.ortumcore.mlir
// RUN: FileCheck %s --check-prefix=SPLIT < %t.ortumcore.mlir
// RUN: ondrix-opt %t.ortumcore.mlir --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --expand-strided-metadata --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --expand-strided-metadata --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: cc %S/Inputs/ondsp_lane_split_aot.c %t.scalar.o -o %t.scalar.bin
// RUN: %t.scalar.bin
// RUN: cc %S/Inputs/ondsp_lane_split_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin

// Three certified chains from aligned and misaligned streams against the
// scalar authority; odd taps exported at shift 1 expose the merge's carry.
// SPLIT-COUNT-3: memref.assume_alignment
// SPLIT-NOT: ondsp.

memref.global "private" constant @split_word_sum_taps : memref<8xi16> = dense<[-1200, 2500, 6000, 9100, 9100, 6000, 2500, -1200]>

func.func @split_word_sum(%x: memref<8xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %table = memref.get_global @split_word_sum_taps : memref<8xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<8xi16>
  %c0 = memref.load %table[%i0] : memref<8xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<8xi16>
  %c1 = memref.load %table[%i1] : memref<8xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<8xi16>
  %c2 = memref.load %table[%i2] : memref<8xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<8xi16>
  %c3 = memref.load %table[%i3] : memref<8xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<8xi16>
  %c4 = memref.load %table[%i4] : memref<8xi16>
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<8xi16>
  %c5 = memref.load %table[%i5] : memref<8xi16>
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i6 = arith.constant 6 : index
  %x6 = memref.load %x[%i6] : memref<8xi16>
  %c6 = memref.load %table[%i6] : memref<8xi16>
  %acc7 = ondsp.mac %acc6, %x6, %c6 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i7 = arith.constant 7 : index
  %x7 = memref.load %x[%i7] : memref<8xi16>
  %c7 = memref.load %table[%i7] : memref<8xi16>
  %acc8 = ondsp.mac %acc7, %x7, %c7 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc8 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

memref.global "private" constant @split_by_halves_taps : memref<8xi16> = dense<[16383, 16381, 16383, 16379, 16383, 16381, 16383, 16379]>

func.func @split_by_halves(%x: memref<8xi16>) -> i32 attributes {llvm.emit_c_interface} {
  %table = memref.get_global @split_by_halves_taps : memref<8xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<8xi16>
  %c0 = memref.load %table[%i0] : memref<8xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<8xi16>
  %c1 = memref.load %table[%i1] : memref<8xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<8xi16>
  %c2 = memref.load %table[%i2] : memref<8xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<8xi16>
  %c3 = memref.load %table[%i3] : memref<8xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<8xi16>
  %c4 = memref.load %table[%i4] : memref<8xi16>
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<8xi16>
  %c5 = memref.load %table[%i5] : memref<8xi16>
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i6 = arith.constant 6 : index
  %x6 = memref.load %x[%i6] : memref<8xi16>
  %c6 = memref.load %table[%i6] : memref<8xi16>
  %acc7 = ondsp.mac %acc6, %x6, %c6 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i7 = arith.constant 7 : index
  %x7 = memref.load %x[%i7] : memref<8xi16>
  %c7 = memref.load %table[%i7] : memref<8xi16>
  %acc8 = ondsp.mac %acc7, %x7, %c7 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc8 {dst = #ondsp.fixed<signed, storage = i32, frac = 29>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<toward_negative>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %r : i32
}

memref.global "private" constant @split_odd_tail_taps : memref<5xi16> = dense<[3000, -2000, 5000, 7000, 1000]>

func.func @split_odd_tail(%x: memref<6xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %table = memref.get_global @split_odd_tail_taps : memref<5xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<6xi16>
  %c0 = memref.load %table[%i0] : memref<5xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<6xi16>
  %c1 = memref.load %table[%i1] : memref<5xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<6xi16>
  %c2 = memref.load %table[%i2] : memref<5xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<6xi16>
  %c3 = memref.load %table[%i3] : memref<5xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<6xi16>
  %c4 = memref.load %table[%i4] : memref<5xi16>
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc5 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}
