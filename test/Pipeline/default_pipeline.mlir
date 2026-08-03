// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" | FileCheck %s
// RUN: ondrix-opt %s --ondrix-default-pipeline | FileCheck %s --check-prefix=SCALAR
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0" | FileCheck %s --check-prefix=SCALAR
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128" | FileCheck %s --check-prefix=NARROW

// The canonical pipeline: contract conversion, boundary bufferization, the
// automatic schedule stage, and lowering to the LLVM dialect — one flag, no
// hand-picked passes. Schedules are selected inside the pipeline: every
// candidate transform is filtered by its own legality analysis and applied in
// the documented priority order, and anything unauthorized keeps its ordered
// scalar schedule. The vector width is a target fact (register bits), not a
// user choice: the same module compiles at any width, and width zero is the
// all-ordered program. Zero is also the default, so an undeclared target gets
// the schedule that is legal everywhere instead of a guess.

// The static f32 filter batches: its lanes carry independent outputs as
// fused-event chains, visible after full lowering as a vector fma.
// CHECK: llvm.func @f32_filter
// CHECK: llvm.intr.fma{{.*}}vector<8xf32>
// NARROW: llvm.intr.fma{{.*}}vector<4xf32>
// SCALAR-NOT: vector<8xf32>
func.func @f32_filter(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                      %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// A site that declared the fast relaxation gets the horizontal reduction the
// exact contracts refuse: fused partial-sum lanes and one cross-lane fadd.
// CHECK: llvm.func @f32_filter_fast
// CHECK: "llvm.intr.vector.reduce.fadd"
// SCALAR: llvm.func @f32_filter_fast
// SCALAR-NOT: vector.reduce.fadd
func.func @f32_filter_fast(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                           %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// A dynamic shape is refused by every schedule candidate and still compiles
// through the same pipeline on the ordered path.
// CHECK: llvm.func @f32_filter_dynamic
func.func @f32_filter_dynamic(%input: tensor<?xf32>, %coeffs: tensor<?xf32>,
                              %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// An algorithm without a bufferized reduction route (the general-window
// moving average) flows through the tensor conversion inside the same
// pipeline and lowers to scalar LLVM arithmetic.
// CHECK: llvm.func @q15_moving_average
// SCALAR: llvm.func @q15_moving_average
func.func @q15_moving_average(%input: tensor<40xi16>) -> tensor<38xi16> {
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<40xi16>) -> tensor<38xi16>
  return %result : tensor<38xi16>
}

// The composed four-stage program exercises every automatic stage at once:
// the design intent is evaluated to its guarded constant table, those
// constants feed the certified saturating horizontal reduction (8 lanes at
// the default width), the staged spectrum reads are forwarded onto the RFFT
// scalars, and width zero keeps the whole chain ordered.
// CHECK: llvm.func @q15_composed_spectrum
// CHECK: vector<8xi16>
// CHECK: vector<8xi64>
// SCALAR: llvm.func @q15_composed_spectrum
func.func @q15_composed_spectrum(%signal: tensor<72xi16>) -> tensor<33xi16> {
  %taps = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  %init = tensor.empty() : tensor<64xi16>
  %filtered = ondrix.fir_filter %signal, %taps, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<72xi16>, tensor<9xi16>, tensor<64xi16>) -> tensor<64xi16>
  %spectrum = ondrix.rfft %filtered {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi16>) -> tensor<33xi32>
  %result = ondrix.cx_magnitude %spectrum {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<33xi32>) -> tensor<33xi16>
  return %result : tensor<33xi16>
}
