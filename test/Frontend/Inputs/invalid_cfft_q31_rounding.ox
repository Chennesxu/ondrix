def q31_cfft8_bad(input: tensor[complex_q31,8]) -> tensor[complex_q31,8]:
  return cfft(input, rounding=nearest_even)
