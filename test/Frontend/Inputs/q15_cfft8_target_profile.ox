def q15_cfft8_floor(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input, rounding=toward_negative, overflow=wrap)
