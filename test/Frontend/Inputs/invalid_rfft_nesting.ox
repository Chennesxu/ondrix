def invalid_rfft_nesting(input: tensor[q15,16]) -> tensor[complex_q15,9]:
  return cfft(rfft(input))
