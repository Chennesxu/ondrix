// RUN: not ondrix-opt %s 2>&1 | FileCheck %s

// The reversed-domain equation is defined on the full 32-bit register.
// CHECK: op operand #0 must be 32-bit signless integer
func.func @narrow_bitrev(%address: i16, %step: i16) -> i16 {
  %next = ortumcore.bitrev_add %address, %step : (i16, i16) -> i16
  return %next : i16
}
