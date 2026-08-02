def invalid_rfft_extent(input: tensor[q15,24]) -> tensor[complex_q15,13]:
  return rfft(input)
