def invalid_f32_magnitude_contract(input: tensor[complex_f32, 8]) -> tensor[f32, 8]:
  return magnitude(input)
