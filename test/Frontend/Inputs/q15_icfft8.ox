def q15_icfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(input)
