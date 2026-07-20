// RUN: ondrix-opt %S/proof_trace.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=8 proof-trace-output=%t.json" > /dev/null
// RUN: not ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.json max-elements=8" 2>&1 | FileCheck %s

memref.global "private" constant @coefficients : memref<8xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -7]>

func.func @safe(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @coefficients : memref<8xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

func.func @safe_second(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @coefficients : memref<8xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// CHECK: error: proof trace record no longer matches this reduction: 0
