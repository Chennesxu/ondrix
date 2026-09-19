// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// An accumulator narrower than the exact product frac would requantize every
// term, which is the contract this operation exists to avoid.
func.func @accumulator_frac_must_be_exact(
    %real: !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
    %lhs: memref<8xi32>, %rhs: memref<8xi32>)
    -> (!ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>) {
  // expected-error @+1 {{accumulator frac 15 does not match the exact product frac 30}}
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
       memref<8xi32>, memref<8xi32>)
      -> (!ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 15, signed, update_overflow = saturate>
}

// -----

// The operand element is the packed container, not one component; accepting a
// component-width memref would silently halve the term count.
func.func @operand_must_be_the_packed_container(
    %real: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  // expected-error @+1 {{packed operand element must be i32 for this layout}}
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

// A split layout has no packed container to walk, so it has no executable
// reduction either.
func.func @layout_must_be_executable(
    %real: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi32>, %rhs: memref<8xi32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  // expected-error @+1 {{requires an executable packed complex layout}}
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<split>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<8xi32>, memref<8xi32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

// Unequal static lengths have no ordered term domain.
func.func @lengths_must_agree(
    %real: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi32>, %rhs: memref<9xi32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  // expected-error @+1 {{shaped operands must have equal static lengths}}
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<8xi32>, memref<9xi32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}
