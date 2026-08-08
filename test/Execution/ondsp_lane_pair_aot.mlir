// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: ondrix-opt %s --convert-ondsp-lane-pairs-to-ortumcore --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/ondsp_lane_pair_aot.c %t.scalar.o -o %t.scalar.bin
// RUN: %t.scalar.bin
// RUN: cc %S/Inputs/ondsp_lane_pair_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin

!pair = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>

// One lanes = 2 source walk, exported twice: the generic scalar authority and
// the dual-lane target composition must agree on every lane and rounding.
func.func @ondsp_pair_walk_floor(%value0: i16, %value1: i16, %coefficient: i16,
                                 %count: index, %lane: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = arith.constant dense<0> : vector<2xi16>
  %low = vector.insert %value0, %empty [0] : i16 into vector<2xi16>
  %values = vector.insert %value1, %low [1] : i16 into vector<2xi16>
  %zero = ondsp.acc_zero : !pair
  %acc = scf.for %i = %c0 to %count step %c1 iter_args(%it = %zero) -> (!pair) {
    %next = ondsp.mac %it, %values, %coefficient {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!pair, vector<2xi16>, i16) -> !pair
    scf.yield %next : !pair
  }
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  %selected = vector.extractelement %out[%lane : i32] : vector<2xi32>
  return %selected : i32
}

func.func @ondsp_pair_walk_ntp(%value0: i16, %value1: i16, %coefficient: i16,
                               %count: index, %lane: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = arith.constant dense<0> : vector<2xi16>
  %low = vector.insert %value0, %empty [0] : i16 into vector<2xi16>
  %values = vector.insert %value1, %low [1] : i16 into vector<2xi16>
  %zero = ondsp.acc_zero : !pair
  %acc = scf.for %i = %c0 to %count step %c1 iter_args(%it = %zero) -> (!pair) {
    %next = ondsp.mac %it, %values, %coefficient {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!pair, vector<2xi16>, i16) -> !pair
    scf.yield %next : !pair
  }
  %out = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (!pair) -> vector<2xi32>
  %selected = vector.extractelement %out[%lane : i32] : vector<2xi32>
  return %selected : i32
}
