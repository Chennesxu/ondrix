// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4" 2>&1 | FileCheck %s

func.func @out_of_range_memory_space(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16, 1267650600228229401496703205376 : i128>,
    %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: error: 'ondsp.reduce_mac' op integer memory space must be nonnegative and fit in an unsigned LLVM address space
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16, 1267650600228229401496703205376 : i128>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}
