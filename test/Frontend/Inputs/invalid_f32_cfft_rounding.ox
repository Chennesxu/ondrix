def invalid_f32_cfft_rounding(input: tensor[complex_f32,16]) -> tensor[complex_f32,16]:
  return cfft(input, contract = off, rounding = nearest_even)
