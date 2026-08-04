def q15_cic_decimate(input: tensor[q15,32]) -> tensor[q15,8]:
  return cic_decimate(input, stages=2, rate=4, delay=1, state_overflow=wrap)
