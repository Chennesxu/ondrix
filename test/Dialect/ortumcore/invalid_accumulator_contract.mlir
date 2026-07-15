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

// -----

func.func @qmac_requires_i32_payload(
    %acc: !ortumcore.acc, %lhs: i16, %rhs: i16) -> !ortumcore.acc {
  // expected-error@+1 {{operand #1 must be 32-bit signless integer}}
  %0 = ortumcore.qmac_add %acc, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}

// -----

func.func @extract_requires_integer_result(%acc: !ortumcore.acc) -> f32 {
  // expected-error@+1 {{result #0 must be 32-bit signless integer}}
  %0 = ortumcore.acc_extract %acc : (!ortumcore.acc) -> f32
  return %0 : f32
}

// -----

func.func @import_requires_integer_input(%input: f32) -> !ortumcore.acc {
  // expected-error@+1 {{operand #0 must be 32-bit signless integer}}
  %0 = ortumcore.acc_import %input : (f32) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}
