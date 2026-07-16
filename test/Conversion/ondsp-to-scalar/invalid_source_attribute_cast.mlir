// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @cast_metadata(%input: i32) -> i32 {
  // CHECK: 'builtin.unrealized_conversion_cast' op attribute 'test.numeric' contains an Ondsp type or attribute
  %result = "builtin.unrealized_conversion_cast"(%input) {
    test.numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i32) -> i32
  return %result : i32
}
