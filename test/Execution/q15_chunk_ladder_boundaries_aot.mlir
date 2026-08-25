// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_chunk_ladder_boundaries_aot.c %t.o -o %t
// RUN: %t

// One extent per interesting position on the width-4 ladder: below one machine
// vector, exactly each rung, and each rung with a tail the chunk cannot cover.

func.func @reduce3(%lhs: memref<3xi16>, %rhs: memref<3xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<3xi16>, memref<3xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce4(%lhs: memref<4xi16>, %rhs: memref<4xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce5(%lhs: memref<5xi16>, %rhs: memref<5xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<5xi16>, memref<5xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce8(%lhs: memref<8xi16>, %rhs: memref<8xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce9(%lhs: memref<9xi16>, %rhs: memref<9xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<9xi16>, memref<9xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce12(%lhs: memref<12xi16>, %rhs: memref<12xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<12xi16>, memref<12xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce13(%lhs: memref<13xi16>, %rhs: memref<13xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<13xi16>, memref<13xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce15(%lhs: memref<15xi16>, %rhs: memref<15xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<15xi16>, memref<15xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce16(%lhs: memref<16xi16>, %rhs: memref<16xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<16xi16>, memref<16xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @reduce17(%lhs: memref<17xi16>, %rhs: memref<17xi16>) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<17xi16>, memref<17xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}
