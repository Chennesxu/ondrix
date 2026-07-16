// RUN: ondrix-opt %s --normalize-ondsp-q15-vector-reduce --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_vector_ordered_aot.c %t.o -o %t
// RUN: %t

func.func @q15_vector_reduce_saturate(
    %lhs: memref<520xi16>, %rhs: memref<520xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c520 = arith.constant 520 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %base = %c0 to %c520 step %c8
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %lhs_vector = vector.load %lhs[%base] : memref<520xi16>, vector<8xi16>
    %rhs_vector = vector.load %rhs[%base] : memref<520xi16>, vector<8xi16>
    %next = ondsp.reduce_mac %current, %lhs_vector, %rhs_vector {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_vector_reduce_wrap(
    %lhs: memref<520xi16>, %rhs: memref<520xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c520 = arith.constant 520 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %base = %c0 to %c520 step %c8
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %lhs_vector = vector.load %lhs[%base] : memref<520xi16>, vector<8xi16>
    %rhs_vector = vector.load %rhs[%base] : memref<520xi16>, vector<8xi16>
    %next = ondsp.reduce_mac %current, %lhs_vector, %rhs_vector {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}
