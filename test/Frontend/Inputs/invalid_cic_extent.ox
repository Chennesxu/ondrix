def invalid_cic_extent(input: tensor[q15,32]) -> tensor[q15,9]:
  return cic_decimate(input, stages=2, rate=4, delay=1, state_overflow=wrap)
