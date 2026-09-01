def q31_cic_decimate(input: tensor[q31,32]) -> tensor[q31,8]:
  return cic_decimate(input, stages=2, rate=4, delay=1, state_overflow=wrap)
