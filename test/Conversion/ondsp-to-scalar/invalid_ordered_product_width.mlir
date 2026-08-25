// RUN: not ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar="vector-width=0" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar="vector-width=-4" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar="vector-width=100000" 2>&1 | FileCheck %s

// The block loop emits one fold per lane, so an unbounded width would be a
// compile-time expansion rather than a schedule.
// CHECK: error: vector-width must be between 1 and 4096

module {
}
