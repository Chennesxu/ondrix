// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @loop_metadata() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: 'scf.for' op attribute 'test.numeric' contains an Ondsp type or attribute
  "scf.for"(%c0, %c1, %c1) ({
  ^bb0(%index: index):
    "scf.yield"() : () -> ()
  }) {test.numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (index, index, index) -> ()
  return
}
