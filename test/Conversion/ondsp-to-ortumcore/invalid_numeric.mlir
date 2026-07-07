// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_unsigned_mac(%acc: !ondsp.acc<storage = i40, frac = 30, signed>,
                                    %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // CHECK: only signed q15 and signed q31 MAC are supported by ortumcore lowering
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<unsigned, storage = i16, frac = 15>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}
