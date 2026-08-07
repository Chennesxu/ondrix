// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/q15_ntp_export_aot.c %t.scalar.o -o %t.scalar.bin
// RUN: %t.scalar.bin
// RUN: cc -DNTP_TARGET_LEG %S/Inputs/q15_ntp_export_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin

// The ties-positive export through the target readout composition must
// agree with the scalar authority and the independent reference, including
// at the saturated positive rail where adding half INTO the accumulator
// would lose the carry.
func.func @ondsp_repeat_mac_ntp(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i32, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %out : i32
}

func.func @ondsp_repeat_mac_ntp_i16(%lhs: i16, %rhs: i16, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}
