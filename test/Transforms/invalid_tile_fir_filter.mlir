// RUN: not ondrix-opt %s --tile-ondrix-fir-filter="tile-size=0" 2>&1 | FileCheck %s

// CHECK: error: tile-size must be positive
module {
}
