def q15_icfft_named_operand(
    cfft: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(cfft)
