// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=0" 2>&1 | FileCheck %s

// CHECK: error: vector-width must be positive

module {
}
