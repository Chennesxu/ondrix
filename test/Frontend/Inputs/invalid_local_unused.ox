def invalid_local_unused(input: tensor[q15,16]) -> tensor[complex_q15,9]:
  taps = lowpass(taps=9, cutoff=[1,4])
  return rfft(input)
