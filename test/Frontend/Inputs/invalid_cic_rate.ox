def invalid_cic_rate(input: tensor[q15,24]) -> tensor[q15,8]:
  return cic_decimate(input, stages=2, rate=3, delay=1, state_overflow=wrap)
