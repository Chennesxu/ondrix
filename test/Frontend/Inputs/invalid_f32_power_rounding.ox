def invalid_f32_power_rounding(input: tensor[complex_f32, 8]) -> tensor[f32, 8]:
  return power(input, rounding=nearest_even)
