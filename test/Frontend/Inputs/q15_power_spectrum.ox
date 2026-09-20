def q15_power_spectrum(input: tensor[q15, 16]) -> tensor[q15, 9]:
  return power(rfft(input))
