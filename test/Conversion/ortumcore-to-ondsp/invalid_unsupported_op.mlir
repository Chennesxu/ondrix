// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @unsupported(%lhs: i32, %rhs: i32) -> i32 {
  // CHECK: failed to legalize operation 'ortumcore.sat_add'
  %sum = ortumcore.sat_add %lhs, %rhs : (i32, i32) -> i32
  return %sum : i32
}
