// RUN: not ondrix-opt %s --tile-ondrix-fir-filter="tile-size=4" 2>&1 | FileCheck %s

// CHECK: error: 'ondrix.fir_filter' op cannot retile an existing FIR output tile
func.func @full_filter(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %c1 = arith.constant 1 : index
  %result = ondrix.fir_filter %input, %coeffs, %init, %c1 {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>, index) -> tensor<?xi16>
  return %result : tensor<?xi16>
}
