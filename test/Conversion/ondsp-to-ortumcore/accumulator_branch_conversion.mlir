// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=NO-SOURCE

func.func @branch_accumulator(%input: !ondsp.acc<storage = i40, frac = 30, signed>)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  cf.br ^forward(%input : !ondsp.acc<storage = i40, frac = 30, signed>)
^forward(%acc: !ondsp.acc<storage = i40, frac = 30, signed>):
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @branch_accumulator(
// CHECK-SAME: %[[INPUT:.*]]: !ortumcore.acc
// CHECK: cf.br ^[[FORWARD:.*]](%[[INPUT]] : !ortumcore.acc)
// CHECK: ^[[FORWARD]](%[[ACC:.*]]: !ortumcore.acc):
// CHECK: return %[[ACC]] : !ortumcore.acc
// NO-SOURCE-NOT: !ondsp.acc
