// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// Unlike the real reduction, the packed complex one walks a container, so a
// tensor operand has no term domain to walk.
func.func @operands_must_be_buffers(
    %lhs: tensor<8xi32>, %rhs: tensor<8xi32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  // expected-error @+1 {{cx_dot operands must be rank-1 memrefs of packed complex values}}
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (tensor<8xi32>, tensor<8xi32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

// The algorithm layer carries the same exact-product contract the numeric
// layer does, so a requantizing accumulator is refused here too.
func.func @accumulator_frac_must_be_exact(
    %lhs: memref<8xi32>, %rhs: memref<8xi32>)
    -> (!ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>) {
  // expected-error @+1 {{accumulator frac 15 does not match the exact product frac 30}}
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (memref<8xi32>, memref<8xi32>)
      -> (!ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>
}
