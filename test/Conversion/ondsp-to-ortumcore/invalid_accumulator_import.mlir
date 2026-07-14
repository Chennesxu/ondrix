// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @acc_import(%input: i16) -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // CHECK: exact accumulator import is unsupported by ortumcore lowering until target import semantics are proven equivalent
  %0 = ondsp.acc_import %input {src = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}
