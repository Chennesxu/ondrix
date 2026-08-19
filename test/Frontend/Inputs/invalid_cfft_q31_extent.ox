def q31_cfft128(input: tensor[complex_q31,128]) -> tensor[complex_q31,128]:
  return cfft(input)
