def q15_cfft8_bad(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input, rounding=toward_zero)
