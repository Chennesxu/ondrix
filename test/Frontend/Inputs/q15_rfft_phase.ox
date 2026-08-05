def q15_rfft_phase(input: tensor[q15,64]) -> tensor[q15,33]:
  return phase(rfft(input))
