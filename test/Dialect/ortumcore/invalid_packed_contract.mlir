// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @packed_op_requires_vector_state(
    %state: i32, %lhs: i32, %rhs: i32) -> (i32, i32) {
  // expected-error@+1 {{operand #0 must be Explicit packed-vector mode/state token}}
  %next, %0 = ortumcore.cx_mul %state, %lhs, %rhs : (i32, i32, i32) -> (i32, i32)
  return %next, %0 : i32, i32
}

// -----

func.func @packed_op_requires_i32_payload(
    %state: !ortumcore.vec.state, %lhs: i8, %rhs: i8)
    -> (!ortumcore.vec.state, i32) {
  // expected-error@+1 {{operand #1 must be 32-bit signless integer}}
  %next, %0 = ortumcore.cx_mul %state, %lhs, %rhs : (!ortumcore.vec.state, i8, i8) -> (!ortumcore.vec.state, i32)
  return %next, %0 : !ortumcore.vec.state, i32
}

// -----

func.func @packed_fft_requires_i32_payload(
    %state: !ortumcore.vec.state, %input: i8)
    -> (!ortumcore.vec.state, i32) {
  // expected-error@+1 {{operand #1 must be 32-bit signless integer}}
  %next, %0 = ortumcore.fft_trivial_stage %state, %input {stage_kind = #ortumcore<fft_stage_kind radix2>} : (!ortumcore.vec.state, i8) -> (!ortumcore.vec.state, i32)
  return %next, %0 : !ortumcore.vec.state, i32
}

// -----

func.func @packed_op_requires_vector_state_result(
    %state: !ortumcore.vec.state, %lhs: i32, %rhs: i32) -> (i32, i32) {
  // expected-error@+1 {{result #0 must be Explicit packed-vector mode/state token}}
  %next, %0 = ortumcore.cx_mul %state, %lhs, %rhs : (!ortumcore.vec.state, i32, i32) -> (i32, i32)
  return %next, %0 : i32, i32
}

// -----

func.func @packed_op_requires_i32_result(
    %state: !ortumcore.vec.state, %lhs: i32, %rhs: i32)
    -> (!ortumcore.vec.state, i8) {
  // expected-error@+1 {{result #1 must be 32-bit signless integer}}
  %next, %0 = ortumcore.cx_mul %state, %lhs, %rhs : (!ortumcore.vec.state, i32, i32) -> (!ortumcore.vec.state, i8)
  return %next, %0 : !ortumcore.vec.state, i8
}

// -----

func.func @packed_fft_requires_i32_result(
    %state: !ortumcore.vec.state, %input: i32)
    -> (!ortumcore.vec.state, i8) {
  // expected-error@+1 {{result #1 must be 32-bit signless integer}}
  %next, %0 = ortumcore.fft_trivial_stage %state, %input {stage_kind = #ortumcore<fft_stage_kind radix2>} : (!ortumcore.vec.state, i32) -> (!ortumcore.vec.state, i8)
  return %next, %0 : !ortumcore.vec.state, i8
}
