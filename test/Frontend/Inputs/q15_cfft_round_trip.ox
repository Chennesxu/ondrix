def q15_cfft_round_trip(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(cfft(input))
