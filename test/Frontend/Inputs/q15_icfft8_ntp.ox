def q15_icfft8_ntp(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(input, rounding=nearest_ties_positive)
