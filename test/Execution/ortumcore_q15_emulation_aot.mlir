// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ortumcore_q15_emulation_aot.c %t.o -o %t
// RUN: %t

func.func @ortumcore_repeat_mac(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_add %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  // This test-only cast observes the raw accumulator bits after emulation
  // expands the capability through Ondsp and scalar finalization maps it to i40.
  %bits = builtin.unrealized_conversion_cast %acc : !ortumcore.acc to i40
  %result = arith.trunci %bits : i40 to i32
  return %result : i32
}

func.func @ortumcore_repeat_mac_sub(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_sub %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  %bits = builtin.unrealized_conversion_cast %acc : !ortumcore.acc to i40
  %result = arith.trunci %bits : i40 to i32
  return %result : i32
}
