// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ortumcore_q15_emulation_aot.c %t.o -o %t
// RUN: %t

func.func @ortumcore_repeat_mac(%lhs: i16, %rhs: i16, %count: index) -> i64 {
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
  %result = arith.extsi %bits : i40 to i64
  return %result : i64
}

func.func @ortumcore_repeat_mac_sub(%lhs: i16, %rhs: i16, %count: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_sub %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  %bits = builtin.unrealized_conversion_cast %acc : !ortumcore.acc to i40
  %result = arith.extsi %bits : i40 to i64
  return %result : i64
}

func.func @ortumcore_repeat_mac_out_q15(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_add %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  %out = ortumcore.acc_out %acc {shift = 15 : i64} : (!ortumcore.acc) -> i32
  return %out : i32
}

func.func @ortumcore_repeat_mac_out_raw(%lhs: i16, %rhs: i16, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_add %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  %out = ortumcore.acc_out %acc {shift = 0 : i64} : (!ortumcore.acc) -> i32
  return %out : i32
}

func.func @ortumcore_bitrev_add_walk(%start: i32, %step: i32, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%address = %start) -> (i32) {
    %next = ortumcore.bitrev_add %address, %step : (i32, i32) -> i32
    scf.yield %next : i32
  }
  return %result : i32
}

func.func @ortumcore_bitrev_sub_walk(%start: i32, %step: i32, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%address = %start) -> (i32) {
    %next = ortumcore.bitrev_sub %address, %step : (i32, i32) -> i32
    scf.yield %next : i32
  }
  return %result : i32
}

func.func @ortumcore_dmac_walk(%lhs0: i16, %rhs0: i16, %lhs1: i16, %rhs1: i16,
                               %count: index, %lane: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %z0 = ortumcore.acc_init : !ortumcore.acc
  %z1 = ortumcore.acc_init : !ortumcore.acc
  %lanes:2 = scf.for %i = %c0 to %count step %c1
      iter_args(%lo = %z0, %hi = %z1) -> (!ortumcore.acc, !ortumcore.acc) {
    %nlo, %nhi = ortumcore.dmac %lo, %hi, %lhs0, %rhs0, %lhs1, %rhs1 : (!ortumcore.acc, !ortumcore.acc, i16, i16, i16, i16) -> (!ortumcore.acc, !ortumcore.acc)
    scf.yield %nlo, %nhi : !ortumcore.acc, !ortumcore.acc
  }
  %out0 = ortumcore.acc_out %lanes#0 {shift = 15 : i64} : (!ortumcore.acc) -> i32
  %out1 = ortumcore.acc_out %lanes#1 {shift = 15 : i64} : (!ortumcore.acc) -> i32
  %pick = arith.cmpi eq, %lane, %c0_i32 : i32
  %out = arith.select %pick, %out0, %out1 : i32
  return %out : i32
}
