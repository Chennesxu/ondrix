// RUN: ondrix-opt %s --specialize-ondrix-constant-fir > %t.specialized.mlir
// RUN: FileCheck %s --check-prefix=SPECIALIZED --implicit-check-not=ondrix.fir < %t.specialized.mlir
// RUN: ondrix-opt %t.specialized.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/constant_fir_specialization_aot.c %t.o -o %t
// RUN: %t

// SPECIALIZED-LABEL: func.func @symmetric_q15_saturate
// SPECIALIZED-NOT: func.func
// SPECIALIZED: ondsp.acc_add_term

memref.global "private" constant @sparse_q15_coefficients : memref<5xi16> =
  dense<[32767, 0, -32768, 0, 12345]>
memref.global "private" constant @symmetric_q15_coefficients : memref<5xi16> =
  dense<[32767, -32768, 12345, -32768, 32767]>
memref.global "private" constant @symmetric_q31_coefficients : memref<4xi32> =
  dense<[2147483647, -2147483648, -2147483648, 2147483647]>
memref.global "private" constant @symmetric_q31_safe_coefficients : memref<4xi32> =
  dense<[1, 2, 2, 1]>

func.func @sparse_q15_saturate(%input: memref<5xi16>) -> i16 {
  %coefficients = memref.get_global @sparse_q15_coefficients : memref<5xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @symmetric_q15_wrap(%input: memref<5xi16>) -> i16 {
  %coefficients = memref.get_global @symmetric_q15_coefficients : memref<5xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @symmetric_q15_saturate(%input: memref<5xi16>) -> i16 {
  %coefficients = memref.get_global @symmetric_q15_coefficients : memref<5xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @symmetric_q15_strided_wrap(
    %input: memref<5xi16, strided<[?], offset: ?>>) -> i16 {
  %coefficients = memref.get_global @symmetric_q15_coefficients : memref<5xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16, strided<[?], offset: ?>>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @symmetric_q31_wrap(%input: memref<4xi32>) -> i32 {
  %coefficients = memref.get_global @symmetric_q31_coefficients : memref<4xi32>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = wrap>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed,
                       update_overflow = wrap>) -> i32
  return %result : i32
}

func.func @symmetric_q31_saturate(%input: memref<4xi32>) -> i32 {
  %coefficients = memref.get_global @symmetric_q31_safe_coefficients : memref<4xi32>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed,
                       update_overflow = saturate>) -> i32
  return %result : i32
}
