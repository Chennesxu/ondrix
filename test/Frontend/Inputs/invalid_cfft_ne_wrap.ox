def q15_cfft8_ne_wrap(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input, rounding=nearest_even, overflow=wrap)
