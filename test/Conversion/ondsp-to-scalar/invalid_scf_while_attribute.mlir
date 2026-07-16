// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @attributed_while(
    %condition: i1,
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: 'scf.while' op attributes on scf.while carrying converted source types are unsupported
  %result = "scf.while"(%initial) ({
  ^bb0(%current: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>):
    "scf.condition"(%condition, %current) : (i1, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> ()
  }, {
  ^bb0(%current: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>):
    "scf.yield"(%current) : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> ()
  }) {test.marker} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}
