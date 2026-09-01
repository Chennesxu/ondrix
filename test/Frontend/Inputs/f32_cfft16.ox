def f32_cfft16(input: tensor[complex_f32,16]) -> tensor[complex_f32,16]:
  return cfft(input, contract = fma)
