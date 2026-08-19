def q31_cfft8(input: tensor[complex_q31,8]) -> tensor[complex_q31,8]:
  return cfft(input)
