// RUN: ondrix-compile %S/Inputs/q15_multi_use_binding.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_multi_use_binding.ox | ondrix-opt --canonicalize --cse | FileCheck %s --check-prefix=COLLAPSED

// Each of the three reads of `t` instantiates the bound call, so the emitted
// tree carries three copies of it before any collapse runs.
// CHECK-LABEL: func.func @q15_multi_use_binding(
// CHECK-COUNT-3: ondrix.mult %arg0, %arg1
// CHECK-NOT: ondrix.mult %arg0, %arg1
// CHECK: ondrix.add

// The duplicated operations are Pure, so cse recovers the single evaluation
// and both remaining reads name it.
// COLLAPSED-LABEL: func.func @q15_multi_use_binding(
// COLLAPSED: %[[T:.*]] = ondrix.mult %arg0, %arg1
// COLLAPSED: %[[SQUARE:.*]] = ondrix.mult %[[T]], %[[T]]
// COLLAPSED: %[[SCALED:.*]] = ondrix.shift %[[SQUARE]]
// COLLAPSED: ondrix.add %[[SCALED]], %[[T]]
// COLLAPSED-NOT: ondrix.mult %arg0, %arg1
