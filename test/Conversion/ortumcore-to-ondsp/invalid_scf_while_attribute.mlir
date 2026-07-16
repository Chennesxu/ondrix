// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

func.func @attributed_while(%condition: i1, %initial: !ortumcore.acc) -> !ortumcore.acc {
  // CHECK: 'scf.while' op attributes on scf.while carrying converted source types are unsupported
  %result = "scf.while"(%initial) ({
  ^bb0(%current: !ortumcore.acc):
    "scf.condition"(%condition, %current) : (i1, !ortumcore.acc) -> ()
  }, {
  ^bb0(%current: !ortumcore.acc):
    "scf.yield"(%current) : (!ortumcore.acc) -> ()
  }) {test.marker} : (!ortumcore.acc) -> !ortumcore.acc
  return %result : !ortumcore.acc
}
