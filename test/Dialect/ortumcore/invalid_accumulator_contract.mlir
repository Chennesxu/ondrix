// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @mac_requires_accumulator(%acc: i32, %lhs: i16, %rhs: i16) -> i32 {
  // expected-error@+1 {{operand #0 must be Saturating Q15 accumulator semantic value}}
  %0 = ortumcore.mac_add %acc, %lhs, %rhs : (i32, i16, i16) -> i32
  return %0 : i32
}

// -----

func.func @mac_requires_integer_payload(
    %acc: !ortumcore.acc, %lhs: f32, %rhs: f32) -> !ortumcore.acc {
  // expected-error@+1 {{operand #1 must be 16-bit signless integer}}
  %0 = ortumcore.mac_add %acc, %lhs, %rhs : (!ortumcore.acc, f32, f32) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}

// -----

func.func @mac_requires_i16_payload(
    %acc: !ortumcore.acc, %lhs: i8, %rhs: i8) -> !ortumcore.acc {
  // expected-error@+1 {{operand #1 must be 16-bit signless integer}}
  %0 = ortumcore.mac_add %acc, %lhs, %rhs : (!ortumcore.acc, i8, i8) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}
