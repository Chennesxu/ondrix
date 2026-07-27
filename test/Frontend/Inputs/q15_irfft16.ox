def q15_irfft16(input: tensor[complex_q15,9]) -> tensor[q15,16]:
  return irfft(input)
