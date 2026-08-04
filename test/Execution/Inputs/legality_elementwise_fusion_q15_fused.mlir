// RUN: true
// Companion module for legality_elementwise_fusion_q15_aot.mlir: the same
// four certified chains under different symbol names, so one harness can
// link the unfused and fused objects together and compare them.

#q15 = #ondsp.fixed<signed, storage = i16, frac = 15>

func.func @fused_negate_wrap_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.negate %0 {numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @fused_shift_directed_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.shift %a {amount = -2 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.shift %0 {amount = -3 : i64, numeric = #q15,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @fused_offset_wrap_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.offset %a {bias = 1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.offset %0 {bias = -1000 : i64, numeric = #q15, overflow = #ondsp.overflow<wrap>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}

func.func @fused_abs_of_negate_pair(%a: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %0 = ondrix.negate %a {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  %1 = ondrix.abs %0 {numeric = #q15, overflow = #ondsp.overflow<saturate>}
    : (tensor<4096xi16>) -> tensor<4096xi16>
  return %1 : tensor<4096xi16>
}
