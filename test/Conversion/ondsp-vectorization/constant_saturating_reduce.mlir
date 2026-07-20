// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=512" | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=7" | FileCheck %s --check-prefix=LIMIT

memref.global "private" constant @safe_q15_coefficients : memref<8xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8]>
memref.global "private" constant @safe_q31_coefficients : memref<4xi32> =
  dense<[1, -2, 3, -4]>
memref.global "private" constant @unsafe_q15_coefficients : memref<512xi16> = dense<-32768>
memref.global "private" @mutable_q15_coefficients : memref<8xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8]>

func.func @safe_q15(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @safe_q15
// CHECK: scf.for
// CHECK: vector.load {{.*}} : memref<8xi16>, vector<4xi16>
// CHECK: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// CHECK: ondsp.acc_add_term
// CHECK-NOT: ondsp.reduce_mac

func.func @safe_q15_full_subview(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %view = memref.subview %coefficients[0] [8] [1]
      : memref<8xi16> to memref<8xi16, strided<[1]>>
  %result = ondrix.fir %input, %view {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16, strided<[1]>>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @safe_q15_full_subview
// CHECK: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// CHECK: ondsp.acc_add_term
// CHECK-NOT: ondsp.reduce_mac

func.func @partial_q15_subview(%input: memref<4xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %view = memref.subview %coefficients[1] [4] [1]
      : memref<8xi16> to memref<4xi16, strided<[1], offset: 1>>
  %result = ondrix.fir %input, %view {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<4xi16>, memref<4xi16, strided<[1], offset: 1>>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @partial_q15_subview
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

func.func @strided_q15_subview(%input: memref<4xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %view = memref.subview %coefficients[0] [4] [2]
      : memref<8xi16> to memref<4xi16, strided<[2]>>
  %result = ondrix.fir %input, %view {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<4xi16>, memref<4xi16, strided<[2]>>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @strided_q15_subview
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

// LIMIT-LABEL: func.func @safe_q15
// LIMIT: ondsp.reduce_mac
// LIMIT-NOT: vector.reduction
// LIMIT-LABEL: func.func @safe_q31

func.func @safe_q31(%input: memref<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q31_coefficients : memref<4xi32>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<4xi32>, memref<4xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @safe_q31
// CHECK: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// CHECK: ondsp.acc_add_term
// CHECK-NOT: ondsp.reduce_mac

func.func @dynamic_q15(%input: memref<?xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @dynamic_q15
// CHECK: cf.assert
// CHECK: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// CHECK-NOT: ondsp.reduce_mac

func.func @nonzero_memory_space_q15(%input: memref<8xi16, 1>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16, 1>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @nonzero_memory_space_q15
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

func.func @unsafe_q15(%input: memref<512xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @unsafe_q15_coefficients : memref<512xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<512xi16>, memref<512xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @unsafe_q15
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

func.func @mutable_q15(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @mutable_q15_coefficients : memref<8xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @mutable_q15
// CHECK: ondsp.reduce_mac

func.func @seeded_q15(%seed: i16, %input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>
  %result = ondsp.reduce_mac %initial, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                   update_overflow = saturate>, memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK-LABEL: func.func @seeded_q15
// CHECK: ondsp.reduce_mac

func.func @wrapping_q15(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = wrap> {
  %coefficients = memref.get_global @safe_q15_coefficients : memref<8xi16>
  %result = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>
}

// CHECK-LABEL: func.func @wrapping_q15
// CHECK: ondsp.reduce_mac
