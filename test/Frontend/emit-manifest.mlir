// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest | FileCheck %s --check-prefix=ORDERED
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest --vector-bits=256 | FileCheck %s --check-prefix=SEPARATE
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest --vector-bits=256 --supports-f32-vector-fma | FileCheck %s --check-prefix=FUSED
// RUN: ondrix-compile %S/Inputs/f32_dot_off.ox --emit=manifest --vector-bits=256 | FileCheck %s --check-prefix=EXACT

// The decisions this compilation made, in the compiler's own terms - not a
// reproduction record: the git revision, the llc invocation, reference compiler
// flags, corpus seeds and object hashes belong to the harness around it, and
// empty fields for them would read as though something had checked them.
//
// The selection sites are the part that cannot be read off the source. Keys sort
// alphabetically, so every prefix below is in document order.

// No declared width: the ordered scalar route, whose fused chain spends F. That
// route is unconditional, so it reports no condition rather than an empty one.
// ORDERED: "vector_bits": 0
// ORDERED: "fast_permissions_used": [
// ORDERED-NEXT: "fuse_multiply_add"
// ORDERED-NEXT: ]
// ORDERED: "mechanism": "ordered_fused"
// ORDERED-NOT: "when"
// ORDERED: "llvm_version": "17.0.6"
// ORDERED: "pipeline": "verify-ondsp-fast-audit-input,evaluate-ondrix-fir-design
// ORDERED: "required_fp_environment"
// ORDERED: "rounding": "round_to_nearest_even"
// ORDERED: "subnormals": "preserved"

// The same source at 256 bits reaches the horizontal route and spends R
// instead, with separate terms. The site carries the condition that selected
// it: a dynamic extent generates this mechanism only where a block exists.
// SEPARATE: "fast_permissions_used": [
// SEPARATE-NEXT: "rebuild_reduction_tree"
// SEPARATE-NEXT: ]
// SEPARATE: "fast_selection_sites": [
// SEPARATE: "instance_domain": "0 <= i < N"
// SEPARATE: "mechanism": "horizontal_separate"
// SEPARATE: "route_role": "whole"
// SEPARATE: "source_operation": "ondsp.reduce_mac"
// SEPARATE: "used_permissions": [
// SEPARATE-NEXT: "rebuild_reduction_tree"
// SEPARATE-NEXT: ]
// SEPARATE: "when": "term_domain >= W"

// Declaring a vector FMA moves that route to both, and generates a second case
// for the branch below one block. The target facts sort first.
// FUSED: "supports_f32_vector_fma": true
// FUSED: "fast_permissions_used": [
// FUSED-NEXT: "fuse_multiply_add"
// FUSED-NEXT: "rebuild_reduction_tree"
// FUSED-NEXT: ]
// Cases sort by their condition, so the short branch is reported first.
// FUSED: "mechanism": "ordered_fused"
// FUSED: "when": "term_domain < W"
// FUSED: "mechanism": "horizontal_fused"
// FUSED: "when": "term_domain >= W"

// An exact contract has no site at all, rather than a site that spent nothing.
// EXACT: "fast_permissions_used": []
// EXACT: "fast_selection_sites": []
