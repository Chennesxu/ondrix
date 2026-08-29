// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true row-horizontal=true" | FileCheck %s

// A foreign occupant of the reserved sibling name with the wrong contents:
// the row-horizontal route must refuse to read it, not stream zeros as the table.
memref.global "private" constant @__ondrix_dct8_f32_rows : memref<8x8xf32> = dense<0.000000e+00>

// CHECK-NOT: memref.get_global @__ondrix_dct8_f32_rows
// CHECK-LABEL: func.func @f32_dct8_fast
// CHECK: memref.get_global @__ondrix_dct8_f32 :
// CHECK: math.fma {{.*}} : vector<4xf32>
// CHECK-NOT: memref.get_global @__ondrix_dct8_f32_rows
func.func @f32_dct8_fast(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
