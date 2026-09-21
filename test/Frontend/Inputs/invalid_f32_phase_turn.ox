def invalid_f32_phase_turn(input: tensor[complex_f32, 8]) -> tensor[f32, 8]:
  return phase(input, turn=q31)
