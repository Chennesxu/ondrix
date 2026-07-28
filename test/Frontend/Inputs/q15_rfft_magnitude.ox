def q15_rfft_magnitude(input: tensor[q15,16]) -> tensor[q15,9]:
  return magnitude(rfft(input))
