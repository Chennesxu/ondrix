// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

func.func private @identity(!ortumcore.acc) -> !ortumcore.acc

func.func @structural(%condition: i1, %lhs: !ortumcore.acc, %rhs: !ortumcore.acc)
    -> !ortumcore.acc {
  %called = func.call @identity(%lhs) : (!ortumcore.acc) -> !ortumcore.acc
  %selected = arith.select %condition, %called, %rhs : !ortumcore.acc
  cf.br ^bb1(%selected : !ortumcore.acc)
^bb1(%branch_value: !ortumcore.acc):
  %if_value = scf.if %condition -> (!ortumcore.acc) {
    scf.yield %branch_value : !ortumcore.acc
  } else {
    scf.yield %rhs : !ortumcore.acc
  }
  %while_value = scf.while (%current = %if_value) : (!ortumcore.acc) -> !ortumcore.acc {
    %false = arith.constant false
    scf.condition(%false) %current : !ortumcore.acc
  } do {
  ^bb0(%current: !ortumcore.acc):
    scf.yield %current : !ortumcore.acc
  }
  return %while_value : !ortumcore.acc
}

// CHECK: func.func private @identity(!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-LABEL: func.func @structural(
// CHECK: call @identity({{.*}}) : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: arith.select {{.*}} : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: cf.br ^bb1({{.*}} : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
// CHECK: ^bb1(%{{.*}}: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>):
// CHECK: scf.if {{.*}} -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
// CHECK: scf.while
// CHECK-NOT: ortumcore
