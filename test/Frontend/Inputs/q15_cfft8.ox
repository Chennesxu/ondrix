def q15_cfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input)
