def invalid_rfft_dynamic(input: tensor[q15]) -> tensor[complex_q15]:
  return rfft(input)
