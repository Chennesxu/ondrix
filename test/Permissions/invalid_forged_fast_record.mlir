// RUN: ondrix-opt %s --verify-ondsp-fast-audit-input -split-input-file -verify-diagnostics

// The audit attributes are ordinary discardable ones, so without this the
// caller could hand the compiler the conclusion it is supposed to reach.

func.func @forged_event_record(%init: f32, %lhs: memref<32xf32>, %rhs: memref<32xf32>) -> f32 {
  // expected-error @+1 {{'ondsp.fast_selection' is a compiler-owned audit attribute}}
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>,
    ondsp.fast_selection = {instance_domain = "0 <= i < N", mechanism = "horizontal_separate",
                            route_role = "whole", source_operation = "ondsp.reduce_mac",
                            source_site_id = "forged/ondsp.reduce_mac#0",
                            used_permissions = ["rebuild_reduction_tree"], when = ""}
  } : (f32, memref<32xf32>, memref<32xf32>) -> f32
  return %r : f32
}

// -----

// expected-error @+1 {{'ondsp.fast_used' is a compiler-owned audit attribute}}
module attributes {ondsp.fast_used = ["rebuild_reduction_tree"]} {
  func.func @forged_module_summary(%input: tensor<8xf32>) -> tensor<8xf32> {
    %r = ondrix.gain %input {
      fp_gain = 2.500000e-01 : f32,
      numeric = #ondsp.fp<format = f32, contract = fast>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}

// -----

// The site stamp is what a later pass inherits, so it is refused too: a forged
// one would attribute this compilation's schedule to a site it never had.
func.func @forged_site_stamp(%init: f32, %lhs: memref<32xf32>, %rhs: memref<32xf32>) -> f32 {
  // expected-error @+1 {{'ondsp.fast_source_site' is a compiler-owned audit attribute}}
  %r = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>,
    ondsp.fast_source_site = {instance_domain = "0 <= g < N", route_role = "full_interior",
                              source_operation = "ondrix.fir_filter",
                              source_site_id = "elsewhere/ondrix.fir_filter#0"}
  } : (f32, memref<32xf32>, memref<32xf32>) -> f32
  return %r : f32
}
