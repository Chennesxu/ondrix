// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" | FileCheck %s --check-prefix=SCHEDULE
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/f32_conv1d_aot.c %t.o -lm -o %t
// RUN: %t

// conv1d was the one f32 operation with no object gate on any contract. Both
// modes are covered because the kernel index order is the whole difference
// between them, and a reversed traversal is exactly the mistake bit-exact
// comparison against an independent reference catches.
//
// fast means different things in the two modes, which is the reason both are
// here. conv1d bufferizes to one `ondsp.reduce_mac` per output over a kernel
// subview: correlation reads it forward at unit stride, so the horizontal
// rewrite accepts it and the reassociation permission is spent; convolution
// reads it at stride -1, which the rewrite refuses, so nothing is spent and
// the declaration is inert.
//
// The two therefore carry different evidence. The inert leg pins the
// lowering's SELECTION bitwise - the fused member is what it builds today, and
// a transform that starts consuming here must redden that rather than change
// the object silently. The consuming leg is a relaxed result and is not
// bit-pinned: it is checked for term conservation on an integer sub-domain
// where every derivable regrouping is exact.
//
// SCHEDULE-LABEL: llvm.func @f32_conv1d_conv_fast
// SCHEDULE-NOT: llvm.intr.vector.reduce.fadd
// SCHEDULE-LABEL: llvm.func @f32_conv1d_corr_fast
// SCHEDULE: llvm.intr.vector.reduce.fadd

func.func @f32_conv1d_conv_off(%input: tensor<12xf32>, %kernel: tensor<4xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

func.func @f32_conv1d_conv_fma(%input: tensor<12xf32>, %kernel: tensor<4xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

func.func @f32_conv1d_conv_fast(%input: tensor<12xf32>, %kernel: tensor<4xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

func.func @f32_conv1d_corr_off(%input: tensor<12xf32>, %kernel: tensor<4xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

func.func @f32_conv1d_corr_fma(%input: tensor<12xf32>, %kernel: tensor<4xf32>) -> tensor<9xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<9xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<12xf32>, tensor<4xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

// Twenty taps at width eight: the lane seed takes the first block, the vector
// loop one more, and the ordered tail four. Twelve taps would not do - the
// seed would cover the only block and the loop body would never run, so a
// mutation inside it could not be caught.
func.func @f32_conv1d_corr_fast(%input: tensor<40xf32>, %kernel: tensor<20xf32>) -> tensor<21xf32>
    attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<21xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<40xf32>, tensor<20xf32>, tensor<21xf32>) -> tensor<21xf32>
  return %result : tensor<21xf32>
}

func.func @f32_conv1d_corr_ordered(%input: tensor<40xf32>, %kernel: tensor<20xf32>)
    -> tensor<21xf32> attributes {llvm.emit_c_interface} {
  %init = tensor.empty() : tensor<21xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<40xf32>, tensor<20xf32>, tensor<21xf32>) -> tensor<21xf32>
  return %result : tensor<21xf32>
}
