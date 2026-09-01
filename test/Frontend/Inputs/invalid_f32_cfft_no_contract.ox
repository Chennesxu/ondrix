def invalid_f32_cfft_no_contract(input: tensor[complex_f32,16]) -> tensor[complex_f32,16]:
  return cfft(input)
