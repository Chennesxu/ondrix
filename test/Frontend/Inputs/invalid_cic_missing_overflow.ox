def invalid_cic_missing_overflow(input: tensor[q15,32]) -> tensor[q15,8]:
  return cic_decimate(input, stages=2, rate=4, delay=1)
