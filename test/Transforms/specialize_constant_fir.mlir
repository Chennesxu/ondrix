// RUN: ondrix-opt %s --specialize-ondrix-constant-fir | FileCheck %s
// RUN: ondrix-opt %s --specialize-ondrix-constant-fir="max-taps=4" | FileCheck %s --check-prefix=LIMIT

memref.global "private" constant @coeff_sparse_q15 : memref<5xi16> = dense<[1, 0, -2, 0, 3]>
memref.global "private" constant @coeff_symmetric_q15 : memref<5xi16> = dense<[1, 2, 0, 2, 1]>
memref.global "private" constant @coeff_symmetric_center_q15 : memref<5xi16> = dense<[1, 2, 3, 2, 1]>
memref.global "private" constant @coeff_symmetric_nonzero_q15 : memref<4xi16> = dense<[1, 2, 2, 1]>
memref.global "private" constant @coeff_zero_q15 : memref<3xi16> = dense<0>
memref.global "private" constant @coeff_symmetric_q31 : memref<4xi32> = dense<[1, 2, 2, 1]>
memref.global "private" constant @coeff_sparse_q31 : memref<4xi32> = dense<[1, 0, 2, 0]>
memref.global "private" constant @coeff_oversized_q15 : memref<65xi16> = dense<0>
memref.global "private" @coeff_mutable_q15 : memref<4xi16> = dense<[1, 2, 2, 1]>

func.func @sparse_q15(
    %input: memref<5xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_sparse_q15 : memref<5xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @sparse_q15
// CHECK-NOT: ondrix.fir
// CHECK-NOT: memref.get_global
// CHECK: ondsp.acc_zero
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK-NOT: ondsp.mac
// LIMIT-LABEL: func.func @sparse_q15
// LIMIT: ondrix.fir

func.func @symmetric_q15_wrap(
    %input: memref<5xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_symmetric_q15 : memref<5xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @symmetric_q15_wrap
// CHECK-NOT: ondrix.fir
// CHECK: arith.extsi {{.*}} : i16 to i17
// CHECK: arith.addi {{.*}} : i17
// CHECK: arith.extsi {{.*}} : i17 to i33
// CHECK: ondsp.acc_add_term {{.*}}term_numeric = #ondsp.fixed<signed, storage = i33, frac = 30>
// CHECK: arith.muli {{.*}} : i33
// CHECK: ondsp.acc_add_term {{.*}}term_numeric = #ondsp.fixed<signed, storage = i33, frac = 30>
// CHECK-NOT: ondsp.mac

func.func @symmetric_q15_nonzero_center(
    %input: memref<5xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_symmetric_center_q15 : memref<5xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @symmetric_q15_nonzero_center
// CHECK-NOT: ondrix.fir
// CHECK-COUNT-2: ondsp.acc_add_term
// CHECK: ondsp.mac

func.func @all_zero_q15(
    %input: memref<3xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_zero_q15 : memref<3xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<3xi16>, memref<3xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @all_zero_q15
// CHECK-NOT: ondrix.fir
// CHECK: ondsp.acc_zero
// CHECK-NOT: memref.load
// CHECK-NOT: ondsp.mac

func.func @symmetric_q31_wrap(
    %input: memref<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_symmetric_q31 : memref<4xi32>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @symmetric_q31_wrap
// CHECK-NOT: ondrix.fir
// CHECK: arith.addi {{.*}} : i33
// CHECK: ondsp.acc_add_term {{.*}}term_numeric = #ondsp.fixed<signed, storage = i65, frac = 62>
// CHECK: arith.muli {{.*}} : i65
// CHECK: ondsp.acc_add_term {{.*}}term_numeric = #ondsp.fixed<signed, storage = i65, frac = 62>
// LIMIT-LABEL: func.func @symmetric_q31_wrap
// LIMIT-NOT: ondrix.fir
// LIMIT: ondsp.acc_add_term

func.func @symmetric_q15_saturate(
    %input: memref<4xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_symmetric_nonzero_q15 : memref<4xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<4xi16>, memref<4xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @symmetric_q15_saturate
// CHECK: ondrix.fir

func.func @symmetric_sparse_q15_saturate(
    %input: memref<5xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_symmetric_q15 : memref<5xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<5xi16>, memref<5xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @symmetric_sparse_q15_saturate
// CHECK-NOT: ondrix.fir
// CHECK-COUNT-4: ondsp.mac
// CHECK-NOT: ondsp.acc_add_term

func.func @symmetric_q31_high_raw(
    %input: memref<4xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_symmetric_q31 : memref<4xi32>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @symmetric_q31_high_raw
// CHECK: ondrix.fir

func.func @sparse_q31_high_raw(
    %input: memref<4xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_sparse_q31 : memref<4xi32>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @sparse_q31_high_raw
// CHECK-NOT: ondrix.fir
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK-NOT: ondsp.mac

func.func @mutable_q15(
    %input: memref<4xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_mutable_q15 : memref<4xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<4xi16>, memref<4xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @mutable_q15
// CHECK: ondrix.fir

func.func @dynamic_input_q15(
    %input: memref<?xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coeffs = memref.get_global @coeff_symmetric_nonzero_q15 : memref<4xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16>, memref<4xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @dynamic_input_q15
// CHECK: ondrix.fir

func.func @oversized_q15(
    %input: memref<65xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coeffs = memref.get_global @coeff_oversized_q15 : memref<65xi16>
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<65xi16>, memref<65xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @oversized_q15
// CHECK: ondrix.fir

func.func @unrelated_arithmetic(%value: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %result = arith.addi %value, %zero : i32
  return %result : i32
}

// CHECK-LABEL: func.func @unrelated_arithmetic
// CHECK: arith.addi
