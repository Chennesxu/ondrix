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

// -----

// The format is the carrier at this profile, so an !ondsp.acc beside it names
// a width the program does not have.
func.func @f32_declares_no_accumulator(%x: memref<16xf32>, %y: memref<16xf32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  // expected-error @+1 {{floating-point cx_dot accumulates in the numeric format and declares no accumulator type of its own}}
  %re, %im = ondrix.cx_dot %x, %y {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (memref<16xf32>, memref<16xf32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @f32_refuses_a_packed_layout(%x: memref<16xf32>, %y: memref<16xf32>) -> (f32, f32) {
  // expected-error @+1 {{floating-point cx_dot requires interleaved layout}}
  %re, %im = ondrix.cx_dot %x, %y {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (memref<16xf32>, memref<16xf32>) -> (f32, f32)
  return %re, %im : f32, f32
}

// -----

// An odd element count names half a complex value at the end of the buffer.
func.func @f32_reads_two_elements_per_value(%x: memref<15xf32>, %y: memref<15xf32>) -> (f32, f32) {
  // expected-error @+1 {{floating-point cx_dot reads two elements per complex value and requires an even operand length of at least two}}
  %re, %im = ondrix.cx_dot %x, %y {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (memref<15xf32>, memref<15xf32>) -> (f32, f32)
  return %re, %im : f32, f32
}
