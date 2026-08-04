// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest | FileCheck %s --check-prefix=ORDERED
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest --vector-bits=256 | FileCheck %s --check-prefix=SEPARATE
// RUN: ondrix-compile %S/Inputs/f32_dot_fast.ox --emit=manifest --vector-bits=256 --supports-f32-vector-fma | FileCheck %s --check-prefix=FUSED
// RUN: ondrix-compile %S/Inputs/f32_dot_off.ox --emit=manifest --vector-bits=256 | FileCheck %s --check-prefix=EXACT

// What this compilation was, in the terms a rerun needs. It records only what
// the compiler owns; the git revision, the llc invocation, reference compiler
// flags, corpus seeds and object hashes belong to the harness around it, and
// empty fields for them would read as though something had checked them.
//
// The permission set is the part that cannot be read off the source: the same
// declaration spends different permissions depending on the target facts, so
// the record has to come from the compilation rather than from the contract.

// No declared width: the ordered scalar route, whose fused chain spends F.
// ORDERED: "vector_bits": 0
// ORDERED: "fast_permissions_used": [
// ORDERED-NEXT: "fuse_multiply_add"
// ORDERED-NEXT: ]

// The same source at 256 bits reaches the horizontal route and spends R
// instead, with separate terms.
// SEPARATE: "fast_permissions_used": [
// SEPARATE-NEXT: "rebuild_reduction_tree"
// SEPARATE-NEXT: ]

// Declaring a vector FMA moves that route to both. The target facts sort first.
// FUSED: "supports_f32_vector_fma": true
// FUSED: "fast_permissions_used": [
// FUSED-NEXT: "fuse_multiply_add"
// FUSED-NEXT: "rebuild_reduction_tree"
// FUSED-NEXT: ]

// An exact contract spends nothing on any route.
// EXACT: "fast_permissions_used": []

// The target it ran for, then the toolchain and the schedule that ran, then the
// environment the numeric model declares and every reference is built to. Keys
// sort alphabetically, so these are in document order.
// ORDERED: "llvm_version": "17.0.6"
// ORDERED: "pipeline": "evaluate-ondrix-fir-design,convert-ondrix-to-ondsp
// ORDERED: "required_fp_environment"
// ORDERED: "rounding": "round_to_nearest_even"
// ORDERED: "subnormals": "preserved"
