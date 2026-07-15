// RUN: not ondrix-opt %s --split-input-file 2>&1 | FileCheck %s

func.func @missing_vec_mode_attributes(%state: !ortumcore.vec.state) {
  // CHECK: requires attribute 'packed_complex'
  %next = ortumcore.vec_set_mode %state : (!ortumcore.vec.state) -> !ortumcore.vec.state
  return
}

// -----

func.func @invalid_vec_mode_operand(%state: i32) {
// CHECK: operand #0 must be Experimental packed-vector mode/state token, but got 'i32'
  %next = ortumcore.vec_set_mode %state {saturation = true, round_to_nearest = false, packed_complex = true, shift_right = 15 : i64, shift_left = 0 : i64} : (i32) -> !ortumcore.vec.state
  return
}

// -----

func.func @invalid_vec_mode_result(%state: !ortumcore.vec.state) {
// CHECK: result #0 must be Experimental packed-vector mode/state token, but got 'i32'
  %next = ortumcore.vec_set_mode %state {saturation = true, round_to_nearest = false, packed_complex = true, shift_right = 15 : i64, shift_left = 0 : i64} : (!ortumcore.vec.state) -> i32
  return
}

// -----

func.func @invalid_vec_mode_shift_type(%state: !ortumcore.vec.state) {
  // CHECK: attribute 'shift_right' failed to satisfy constraint
  %next = ortumcore.vec_set_mode %state {saturation = true, round_to_nearest = false, packed_complex = true, shift_right = 15 : i32, shift_left = 0 : i64} : (!ortumcore.vec.state) -> !ortumcore.vec.state
  return
}

// -----

func.func @invalid_vec_state_init_result() {
  // CHECK: custom op 'ortumcore.vec_state_init' invalid kind of type specified
  %state = ortumcore.vec_state_init : i32
  return
}

// -----

func.func @invalid_vec_mode_rounding_type(%state: !ortumcore.vec.state) {
  // CHECK: attribute 'round_to_nearest' failed to satisfy constraint
  %next = ortumcore.vec_set_mode %state {saturation = true, round_to_nearest = 0 : i64, packed_complex = true, shift_right = 15 : i64, shift_left = 0 : i64} : (!ortumcore.vec.state) -> !ortumcore.vec.state
  return
}
